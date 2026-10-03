"""Checkpoint / auto-resume for the ARS/SAC training loop (train_ars.py).

Layout (one rolling checkpoint + one safety predecessor):

    <ckpt_dir>/
      latest/            <- complete checkpoint (atomic dir swap)
        meta.json        iteration, wandb run id, rew_scale, buffer counters,
                         np.random state, config fingerprint
        learner.msgpack  flax to_bytes of {actor,critic,target_critic,alpha,rng}
                         (params + opt_state + step per model; apply_fn/tx are
                         rebuilt from config on restore)
        buffer.npz       filled slice [:, :size] of every replay-buffer array
      prev/              <- previous "latest" (kept in case latest is torn by
                            a mid-rename kill; pruned on the next save)

Atomicity: everything is written into a tmp dir first, then swapped in with
os.rename, so a SIGKILL mid-save can never corrupt an existing checkpoint.
Stale ``.tmp.*`` dirs from killed saves (any pid) are swept at the start of
the next save.  ``prev`` is pruned BEFORE the tmp write so peak disk usage is
~2x one checkpoint (latest + tmp), not 3x; ``latest`` stays intact throughout.

Single-writer assumption: one process per checkpoint dir. Two jobs sharing a
dir (double-submitted grid entry) can race the rotation — don't do that.

The environment simulator state is NOT checkpointed (GPU sim state
is not portable); resume does a fresh env.reset() and continues from the
saved iteration.  This mirrors the PPO GPU path's resume semantics.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import flax.serialization
import jax
import numpy as np

META_NAME = "meta.json"
LEARNER_NAME = "learner.msgpack"
BUFFER_NAME = "buffer.npz"

_BUFFER_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "masks",
    "dones_float",
    "next_observations",
)


def _learner_state(agent) -> Dict[str, Any]:
    """Serializable pytree of the Learner's mutable state.

    ``MultiModelPCGrad`` carries a per-model ``rng`` advanced by every
    projection step — include it when present so ``use_pcgrad=True`` resumes
    faithfully (plain ``MultiModel`` has no such field).
    """

    def model_state(m):
        s = {"step": m.step, "params": m.params, "opt_state": m.opt_state}
        if getattr(m, "rng", None) is not None:
            s["model_rng"] = m.rng
        return s

    state = {
        "actor": model_state(agent.actor),
        "critic": model_state(agent.critic),
        "target_critic": {
            "step": agent.target_critic.step,
            "params": agent.target_critic.params,
        },
        "alpha": model_state(agent.alpha),
        "rng": agent.rng,
    }
    # PopArt running stats (critic-balancing arms); constant-initial for the
    # non-popart arms — tiny either way, keeps the blob schema uniform.
    if getattr(agent, "popart_state", None) is not None:
        state["popart"] = agent.popart_state
    return state


def restore_learner(agent, blob: bytes):
    """Load serialized state into a freshly constructed Learner (in place).

    The fresh Learner provides the pytree template (shapes/dtypes and the
    non-serializable apply_fn/tx), so from_bytes validates structure for us.
    """
    template = _learner_state(agent)
    state = flax.serialization.from_bytes(template, blob)
    dev = jax.tree_util.tree_map(np.asarray, state)  # host -> device lazily on use

    def _model_kwargs(name):
        kw = {
            "step": int(dev[name]["step"]),
            "params": dev[name]["params"],
            "opt_state": dev[name]["opt_state"],
        }
        if "model_rng" in dev[name]:  # MultiModelPCGrad per-model rng
            kw["rng"] = jax.numpy.asarray(dev[name]["model_rng"])
        return kw

    agent.actor = agent.actor.replace(**_model_kwargs("actor"))
    agent.critic = agent.critic.replace(**_model_kwargs("critic"))
    agent.target_critic = agent.target_critic.replace(
        step=int(dev["target_critic"]["step"]),
        params=dev["target_critic"]["params"],
    )
    agent.alpha = agent.alpha.replace(**_model_kwargs("alpha"))
    agent.rng = jax.numpy.asarray(dev["rng"])
    if "popart" in dev and getattr(agent, "popart_state", None) is not None:
        agent.popart_state = jax.tree_util.tree_map(
            jax.numpy.asarray, dev["popart"]
        )
    agent.models = {"alpha": agent.alpha, "actor": agent.actor, "critic": agent.critic}
    return agent


def _np_random_state_to_json() -> list:
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return [name, keys.tolist(), int(pos), int(has_gauss), float(cached)]


def _np_random_state_from_json(s) -> None:
    np.random.set_state((s[0], np.array(s[1], dtype=np.uint32), s[2], s[3], s[4]))


def save_checkpoint(
    ckpt_dir: str,
    *,
    iteration: int,
    config_fingerprint: str,
    wandb_run_id: str,
    agent,
    replay_buffer,
    rew_scale: Optional[np.ndarray],
    save_buffer: bool = True,
) -> str:
    """Write a complete checkpoint atomically; returns the checkpoint path."""
    os.makedirs(ckpt_dir, exist_ok=True)
    # Sweep stale tmp dirs from ANY previous (killed) save — pids differ across
    # restarts, so match the prefix, not our own pid. Single-writer assumption.
    for stale in glob.glob(os.path.join(ckpt_dir, ".tmp.*")):
        shutil.rmtree(stale, ignore_errors=True)
    # Prune prev BEFORE writing tmp: peak disk = latest + tmp (~2x), and
    # `latest` remains valid at every instant of the sequence.
    prev = os.path.join(ckpt_dir, "prev")
    if os.path.isdir(prev):
        shutil.rmtree(prev)
    tmp = os.path.join(ckpt_dir, f".tmp.{os.getpid()}")
    os.makedirs(tmp)

    meta = {
        "iteration": int(iteration),
        "config_fingerprint": config_fingerprint,
        "git_rev": _git_rev(),  # informational; resume warns (not fails) on drift
        "wandb_run_id": wandb_run_id,
        "agent_initialized": agent is not None,
        "rew_scale": None if rew_scale is None else np.asarray(rew_scale).tolist(),
        "buffer_size": int(replay_buffer.size),
        "buffer_insert_index": int(replay_buffer.insert_index),
        "buffer_capacity": int(replay_buffer.capacity),
        "buffer_saved": bool(save_buffer),
        "np_random_state": _np_random_state_to_json(),
    }
    with open(os.path.join(tmp, META_NAME), "w") as f:
        json.dump(meta, f)

    if agent is not None:
        host = jax.device_get(_learner_state(agent))
        with open(os.path.join(tmp, LEARNER_NAME), "wb") as f:
            f.write(flax.serialization.to_bytes(host))

    if save_buffer:
        size = int(replay_buffer.size)
        arrays = {
            name: getattr(replay_buffer, name)[:, :size] for name in _BUFFER_FIELDS
        }
        # uncompressed savez: the buffer is multi-GB and zlib would take
        # minutes; disk is the cheap resource here.
        np.savez(os.path.join(tmp, BUFFER_NAME), **arrays)

    latest = os.path.join(ckpt_dir, "latest")
    if os.path.isdir(latest):
        os.rename(latest, prev)
    os.rename(tmp, latest)
    return latest


def offload_checkpoint(ckpt_path: str) -> bool:
    """Upload a completed checkpoint to the CHECKPOINT_REMOTE rclone remote; drop local copies on success."""
    if os.environ.get("CHECKPOINT_OFFLOAD", "0") not in {"1", "true", "True"}:
        return False
    remote_root = os.environ.get("CHECKPOINT_REMOTE", "").rstrip("/")
    if not remote_root:
        print("[checkpoint] offload enabled but CHECKPOINT_REMOTE is empty; keeping local")
        return False
    ckpt_path = os.path.abspath(ckpt_path)
    parent = os.path.basename(os.path.dirname(ckpt_path))
    group = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    name = os.path.basename(ckpt_path)
    remote = f"{remote_root}/{group}/{parent}/{name}"
    try:
        timeout = int(os.environ.get("CHECKPOINT_RCLONE_TIMEOUT", "21600"))
        args = ["rclone", "copy", ckpt_path, remote, "--transfers", "1", "--checkers", "2", "--retries", "3"]
        subprocess.run(args, check=True, timeout=timeout)
        prev = os.path.join(os.path.dirname(ckpt_path), "prev")
        if os.path.isdir(prev):
            subprocess.run(["rclone", "copy", prev, f"{remote_root}/{group}/{parent}/prev", "--transfers", "1", "--checkers", "2", "--retries", "3"], check=True, timeout=timeout)
            shutil.rmtree(prev)
        shutil.rmtree(ckpt_path)
        print(f"[checkpoint] offloaded {ckpt_path} -> {remote}; local copies removed")
        return True
    except Exception as e:
        print(f"[checkpoint] offload failed ({e}); retaining local checkpoint")
        return False


def _is_complete(d: str) -> bool:
    """A checkpoint is complete when meta.json exists AND every file it
    promises (learner state, buffer) is present. Guards against truncated
    node-crash leftovers being picked as the resume source."""
    meta_p = os.path.join(d, META_NAME)
    if not os.path.isfile(meta_p):
        return False
    try:
        with open(meta_p) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if meta.get("agent_initialized") and not os.path.isfile(
        os.path.join(d, LEARNER_NAME)
    ):
        return False
    if meta.get("buffer_saved") and meta.get("buffer_size", 0) > 0 and not os.path.isfile(
        os.path.join(d, BUFFER_NAME)
    ):
        return False
    return True


def _git_rev() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def current_git_rev() -> str:
    return _git_rev()


def resumable_candidates(ckpt_dir: str) -> List[str]:
    """Complete checkpoint dirs under ckpt_dir, newest first (latest, prev)."""
    return [
        d
        for cand in ("latest", "prev")
        if _is_complete(d := os.path.join(ckpt_dir, cand))
    ]


def find_resumable(ckpt_dir: str) -> Optional[str]:
    """Return the newest complete checkpoint dir under ckpt_dir, or None.

    Falls back from ``latest`` to ``prev`` when latest is incomplete (e.g. a
    node crash tore the fs state despite the rename protocol)."""
    cands = resumable_candidates(ckpt_dir)
    return cands[0] if cands else None


def load_meta(ckpt_path: str) -> Dict[str, Any]:
    with open(os.path.join(ckpt_path, META_NAME)) as f:
        return json.load(f)


def load_buffer(ckpt_path: str, replay_buffer, meta: Dict[str, Any]) -> None:
    """Restore the replay buffer arrays + counters in place."""
    if not meta.get("buffer_saved", False):
        return
    size = int(meta["buffer_size"])
    if size == 0:
        return
    with np.load(os.path.join(ckpt_path, BUFFER_NAME)) as data:
        for name in _BUFFER_FIELDS:
            dst = getattr(replay_buffer, name)
            src = data[name]
            if dst.shape[0] != src.shape[0] or dst.shape[1] < size:
                raise ValueError(
                    f"checkpoint buffer {name} shape {src.shape} incompatible "
                    f"with allocated buffer {dst.shape} (size={size}); did the "
                    "capacity or task set change between runs?"
                )
            dst[:, :size] = src
    replay_buffer.size = size
    replay_buffer.insert_index = int(meta["buffer_insert_index"]) % replay_buffer.capacity


def restore_np_random(meta: Dict[str, Any]) -> None:
    s = meta.get("np_random_state")
    if s:
        _np_random_state_from_json(s)


def learner_blob(ckpt_path: str) -> Optional[bytes]:
    p = os.path.join(ckpt_path, LEARNER_NAME)
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as f:
        return f.read()
