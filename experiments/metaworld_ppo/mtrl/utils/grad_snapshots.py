"""Per-task gradient snapshot writer + offline loader / replay utilities.

Produces a directory of ``.npz`` snapshots and a single ``manifest.json`` so
analysis scripts can reconstruct K×K Gram matrices, eigenspectra, per-layer
breakdowns, offline FairGrad solutions for arbitrary ``α``, and the PopArt
``σ⁻²`` inversion used to recover what gradients *would* have been under
vanilla critic loss on the same model state.

Design:

- The writer is called once per gated minibatch from
  :meth:`MultiTaskPPO._update_with_batch`.  It receives the per-task flat
  gradient list cached on the per-side optimizer (one element per task, each
  a flat tensor of length ``partition.all_total_numel``), slices each into
  the actor view + critic-full view via the existing scope helpers, and
  writes both views in fp32 alongside per-task σ/μ/W/b (PopArt), per-task
  FairGrad weights/contributions if available, per-task long-EMA success rate
  for alive/dead labeling, and minibatch task counts.

- A single ``manifest.json`` is written on the first dump call.  It records
  task names, layer-level offsets within the actor and critic views (so the
  loader can slice out per-layer grads), and PopArt enabled flag.

- The loader returns a structured object with one xarray-like indexable
  field per metric.  PopArt inversion is exposed as a method that takes a
  Gram matrix and the σ vector at that snapshot and returns the
  vanilla-equivalent Gram.  Offline FairGrad replay calls back into the
  production solver in :mod:`mtrl.gradients.fairgrad` so the figures use
  the exact production code path.

Storage budget (MT50, 50 tasks, ~683k backbone params):
  ~137 MB / snapshot fp32 → keep ``dump_grads_every_collects`` ≥ 5 in
  long runs.  fp16 is rejected: dead-task ‖g‖ ≈ 1e-22 underflows fp16.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------


MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1


def _layer_slices_from_param_seq(
    param_seq: Sequence[torch.nn.Parameter],
    name_by_id: Dict[int, str],
    offsets_by_id: Dict[int, Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """Build the per-layer slice list for one view (actor or critic).

    Each entry: ``{"name": "<dotted.path>", "shape": [...], "offset": int,
    "numel": int}``.  ``offset`` is into the *view* slice, not the global
    flat grad.  Order matches ``param_seq`` (which is filtered to the view).
    """
    out: List[Dict[str, Any]] = []
    for p in param_seq:
        pid = id(p)
        offset, numel = offsets_by_id[pid]
        out.append({
            "name": name_by_id.get(pid, f"unknown_id_{pid}"),
            "shape": list(p.shape),
            "offset": int(offset),
            "numel": int(numel),
        })
    return out


def write_manifest(
    out_dir: str,
    *,
    task_names: List[str],
    actor_layers: List[Dict[str, Any]],
    critic_layers: List[Dict[str, Any]],
    actor_total_numel: int,
    critic_total_numel: int,
    use_popart: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write the dump-directory manifest.  Idempotent (overwrites)."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "version": MANIFEST_VERSION,
        "task_names": list(task_names),
        "num_tasks": len(task_names),
        "actor": {
            "total_numel": int(actor_total_numel),
            "layers": actor_layers,
        },
        "critic": {
            "total_numel": int(critic_total_numel),
            "layers": critic_layers,
        },
        "use_popart": bool(use_popart),
    }
    if extra:
        manifest.update(extra)
    with open(os.path.join(out_dir, MANIFEST_FILENAME), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Writer (called from MultiTaskPPO during training)
# ---------------------------------------------------------------------------


def write_snapshot(
    out_dir: str,
    *,
    collect_idx: int,
    repeat_idx: int,
    minibatch_idx: int,
    gradient_step: int,
    task_names_in_batch: List[str],
    actor_grads: torch.Tensor,         # (K_in_batch, actor_total_numel) fp32
    critic_full_grads: torch.Tensor,   # (K_in_batch, critic_total_numel) fp32
    task_count_in_minibatch: np.ndarray,  # (num_tasks_total,) int
    popart_sigma: Optional[np.ndarray] = None,  # (num_tasks_total,) fp64
    popart_mu: Optional[np.ndarray] = None,
    popart_W: Optional[np.ndarray] = None,
    popart_b: Optional[np.ndarray] = None,
    fairgrad_actor_weights: Optional[np.ndarray] = None,   # (K_in_batch,)
    fairgrad_critic_weights: Optional[np.ndarray] = None,
    fairgrad_actor_residual: Optional[float] = None,
    fairgrad_critic_residual: Optional[float] = None,
    long_ema_success: Optional[np.ndarray] = None,         # (num_tasks_total,)
    value_error_per_task: Optional[np.ndarray] = None,     # (num_tasks_total,)
) -> str:
    """Atomically write one snapshot npz.  Returns the path written.

    File layout: ``<out_dir>/snap_collect{cc:06d}_step{gs:08d}.npz``.  Atomic
    via ``tempfile + rename`` so a partial write from a crashed run is
    detectable (no .tmp left over after a clean save).
    """
    fname = f"snap_collect{collect_idx:06d}_step{gradient_step:08d}.npz"
    path = os.path.join(out_dir, fname)
    # ``np.savez_compressed`` auto-appends ".npz" when the path has no
    # extension, so use an explicit path that already ends in .npz to keep
    # the filename ours and avoid surprise concatenation.
    tmp_path = path + ".tmp.npz"

    payload: Dict[str, np.ndarray] = {
        "collect_idx": np.asarray(collect_idx, dtype=np.int64),
        "repeat_idx": np.asarray(repeat_idx, dtype=np.int64),
        "minibatch_idx": np.asarray(minibatch_idx, dtype=np.int64),
        "gradient_step": np.asarray(gradient_step, dtype=np.int64),
        "task_names_in_batch": np.asarray(task_names_in_batch),
        "actor_grads": actor_grads.detach().to(dtype=torch.float32, device="cpu").numpy(),
        "critic_full_grads": critic_full_grads.detach().to(dtype=torch.float32, device="cpu").numpy(),
        "task_count_in_minibatch": np.asarray(task_count_in_minibatch, dtype=np.int64),
    }
    if popart_sigma is not None:
        payload["popart_sigma"] = np.asarray(popart_sigma, dtype=np.float64)
        payload["popart_mu"] = np.asarray(popart_mu, dtype=np.float64)
        payload["popart_W"] = np.asarray(popart_W, dtype=np.float64)
        payload["popart_b"] = np.asarray(popart_b, dtype=np.float64)
    if fairgrad_actor_weights is not None:
        payload["fairgrad_actor_weights"] = np.asarray(fairgrad_actor_weights, dtype=np.float64)
    if fairgrad_critic_weights is not None:
        payload["fairgrad_critic_weights"] = np.asarray(fairgrad_critic_weights, dtype=np.float64)
    if fairgrad_actor_residual is not None:
        payload["fairgrad_actor_residual"] = np.asarray(fairgrad_actor_residual, dtype=np.float64)
    if fairgrad_critic_residual is not None:
        payload["fairgrad_critic_residual"] = np.asarray(fairgrad_critic_residual, dtype=np.float64)
    if long_ema_success is not None:
        payload["long_ema_success"] = np.asarray(long_ema_success, dtype=np.float64)
    if value_error_per_task is not None:
        payload["value_error_per_task"] = np.asarray(value_error_per_task, dtype=np.float64)

    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, path)
    return path


# ---------------------------------------------------------------------------
# Loader / offline replay
# ---------------------------------------------------------------------------


@dataclass
class GradSnapshot:
    """One loaded snapshot.  See :func:`load_snapshot` and :class:`SnapshotSeries`."""
    path: str
    collect_idx: int
    gradient_step: int
    task_names_in_batch: List[str]      # length K_in_batch (≤ num_tasks_total)
    actor_grads: np.ndarray             # (K_in_batch, actor_numel) fp32
    critic_full_grads: np.ndarray       # (K_in_batch, critic_numel) fp32
    task_count_in_minibatch: np.ndarray # (num_tasks_total,) int
    popart_sigma: Optional[np.ndarray] = None
    popart_mu: Optional[np.ndarray] = None
    popart_W: Optional[np.ndarray] = None
    popart_b: Optional[np.ndarray] = None
    fairgrad_actor_weights: Optional[np.ndarray] = None
    fairgrad_critic_weights: Optional[np.ndarray] = None
    fairgrad_actor_residual: Optional[float] = None
    fairgrad_critic_residual: Optional[float] = None
    long_ema_success: Optional[np.ndarray] = None
    value_error_per_task: Optional[np.ndarray] = None

    # ----- Gram matrix helpers -----

    def actor_gram(self) -> np.ndarray:
        """K×K Gram on the actor view, fp64."""
        g = self.actor_grads.astype(np.float64)
        return g @ g.T

    def critic_gram(self) -> np.ndarray:
        """K×K Gram on the critic-full view, fp64."""
        g = self.critic_full_grads.astype(np.float64)
        return g @ g.T

    def task_indices_in_batch(self, all_task_names: List[str]) -> np.ndarray:
        """Indices into the global task_names of the K tasks present this minibatch."""
        name_to_id = {n: i for i, n in enumerate(all_task_names)}
        return np.asarray([name_to_id[n] for n in self.task_names_in_batch], dtype=np.int64)

    def critic_gram_vanilla_inverted(
        self, all_task_names: List[str]
    ) -> Optional[np.ndarray]:
        """Return ``G_vanilla[i,j] = σ_i² σ_j² · G_norm[i,j]`` on the critic view.

        Derivation: under PopArt the per-trunk gradient is
        ``g_norm = g_vanilla / σ²`` because the normalized critic loss is
        ``L_norm = (R - V_raw)² / σ²``.  Therefore Gram entries gain
        ``σ_i² σ_j²`` when inverting back to the vanilla regime.

        Returns None when the snapshot was not taken under PopArt
        (``popart_sigma is None``); in that case the existing
        :meth:`critic_gram` already *is* the vanilla Gram.
        """
        if self.popart_sigma is None:
            return None
        idx = self.task_indices_in_batch(all_task_names)
        sigma_in_batch = self.popart_sigma[idx].astype(np.float64)
        G_norm = self.critic_gram()
        return G_norm * np.outer(sigma_in_batch ** 2, sigma_in_batch ** 2)

    # ----- Layer slicing (uses manifest) -----

    def actor_layer(self, layer_name: str, manifest: "Manifest") -> np.ndarray:
        """Slice a (K_in_batch, layer_numel) sub-grad from the actor view."""
        layer = manifest.find_actor_layer(layer_name)
        return self.actor_grads[:, layer["offset"]: layer["offset"] + layer["numel"]]

    def critic_layer(self, layer_name: str, manifest: "Manifest") -> np.ndarray:
        """Slice a (K_in_batch, layer_numel) sub-grad from the critic view."""
        layer = manifest.find_critic_layer(layer_name)
        return self.critic_full_grads[:, layer["offset"]: layer["offset"] + layer["numel"]]

    # ----- Offline FairGrad replay (uses production solver) -----

    def replay_fairgrad(
        self,
        side: str,
        alpha: float,
    ) -> "FairGradReplay":
        """Solve FairGrad on this snapshot's grads with arbitrary α.

        ``side`` ∈ {"actor", "critic"}.  Calls the production solver in
        :mod:`mtrl.gradients.fairgrad` so the offline replay matches what
        production would have produced on the same K×D matrix.
        """
        from mtrl.gradients.fairgrad import solve_fairgrad_weights

        grads = self.actor_grads if side == "actor" else self.critic_full_grads
        gt = torch.from_numpy(grads.astype(np.float64))
        diag = solve_fairgrad_weights(gt, alpha=alpha)
        return FairGradReplay(
            side=side,
            alpha=alpha,
            weights=diag.weights.detach().cpu().numpy(),
            weight_entropy=diag.weight_entropy,
            gram_trace=diag.gram_trace,
            residual=diag.residual,
            solver_iters=diag.solver_iters,
            used_scipy_fallback=diag.used_scipy_fallback,
            solver_fallback=diag.solver_fallback,
        )


@dataclass
class FairGradReplay:
    """Output of :meth:`GradSnapshot.replay_fairgrad`."""
    side: str
    alpha: float
    weights: np.ndarray  # (K_in_batch,)
    weight_entropy: float
    gram_trace: float
    residual: float
    solver_iters: int
    used_scipy_fallback: bool
    solver_fallback: bool


@dataclass
class Manifest:
    """Loaded manifest with helper accessors."""
    version: int
    task_names: List[str]
    num_tasks: int
    actor_total_numel: int
    critic_total_numel: int
    actor_layers: List[Dict[str, Any]]
    critic_layers: List[Dict[str, Any]]
    use_popart: bool
    raw: Dict[str, Any]

    def find_actor_layer(self, name: str) -> Dict[str, Any]:
        for L in self.actor_layers:
            if L["name"] == name:
                return L
        raise KeyError(f"actor layer {name!r} not in manifest; available: {[L['name'] for L in self.actor_layers]}")

    def find_critic_layer(self, name: str) -> Dict[str, Any]:
        for L in self.critic_layers:
            if L["name"] == name:
                return L
        raise KeyError(f"critic layer {name!r} not in manifest; available: {[L['name'] for L in self.critic_layers]}")


def load_manifest(path: str) -> Manifest:
    """Load a snapshot directory's ``manifest.json``."""
    if os.path.isdir(path):
        path = os.path.join(path, MANIFEST_FILENAME)
    with open(path, "r") as f:
        raw = json.load(f)
    return Manifest(
        version=int(raw.get("version", 1)),
        task_names=list(raw["task_names"]),
        num_tasks=int(raw["num_tasks"]),
        actor_total_numel=int(raw["actor"]["total_numel"]),
        critic_total_numel=int(raw["critic"]["total_numel"]),
        actor_layers=list(raw["actor"]["layers"]),
        critic_layers=list(raw["critic"]["layers"]),
        use_popart=bool(raw.get("use_popart", False)),
        raw=raw,
    )


def load_snapshot(path: str) -> GradSnapshot:
    """Load one snapshot npz."""
    with np.load(path, allow_pickle=False) as data:
        kwargs: Dict[str, Any] = {
            "path": path,
            "collect_idx": int(data["collect_idx"]),
            "gradient_step": int(data["gradient_step"]),
            "task_names_in_batch": [str(s) for s in data["task_names_in_batch"]],
            "actor_grads": np.asarray(data["actor_grads"]),
            "critic_full_grads": np.asarray(data["critic_full_grads"]),
            "task_count_in_minibatch": np.asarray(data["task_count_in_minibatch"]),
        }
        for opt_key in (
            "popart_sigma", "popart_mu", "popart_W", "popart_b",
            "fairgrad_actor_weights", "fairgrad_critic_weights",
            "long_ema_success", "value_error_per_task",
        ):
            if opt_key in data.files:
                kwargs[opt_key] = np.asarray(data[opt_key])
        for opt_scalar in ("fairgrad_actor_residual", "fairgrad_critic_residual"):
            if opt_scalar in data.files:
                kwargs[opt_scalar] = float(np.asarray(data[opt_scalar]))
    return GradSnapshot(**kwargs)


def list_snapshots(out_dir: str) -> List[str]:
    """Return absolute paths to all ``snap_*.npz`` in ``out_dir``, sorted."""
    if not os.path.isdir(out_dir):
        return []
    files = [
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith("snap_") and f.endswith(".npz")
    ]
    files.sort()
    return files


@dataclass
class SnapshotSeries:
    """Lazy iterator over all snapshots in a directory.  Pairs with a Manifest."""
    out_dir: str
    manifest: Manifest

    def __iter__(self):
        for path in list_snapshots(self.out_dir):
            yield load_snapshot(path)

    def __len__(self) -> int:
        return len(list_snapshots(self.out_dir))

    def collect_indices(self) -> List[int]:
        """Return collect_idx for each snapshot in load order (cheap — only opens header)."""
        out: List[int] = []
        for path in list_snapshots(self.out_dir):
            with np.load(path, allow_pickle=False) as data:
                out.append(int(data["collect_idx"]))
        return out


def open_snapshot_dir(out_dir: str) -> SnapshotSeries:
    """Open a snapshot dir: load manifest + return iterable over snapshots."""
    manifest = load_manifest(out_dir)
    return SnapshotSeries(out_dir=out_dir, manifest=manifest)
