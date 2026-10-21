import os
import inspect

import json
import uuid
import numpy as np
import tqdm
import math
import pyrallis
import wandb
from dataclasses import dataclass, asdict

from dataset_utils import MultiTaskReplayBuffer
import utils

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Trainer-side checkpoint extras for the LoRA arm.  checkpoint_utils (shared
# with train_ars.py, untouched) only serializes the BASE
# actor/critic/target_critic/alpha models; the LoRA-phase state — attached
# flag, attach-time scale, the lora_{actor,critic,target_critic} models and
# the eval-success vector driving the reset-vs-attach decision — is written
# by this trainer into the completed checkpoint dir as side files.
LORA_META_NAME = "lora_meta.json"
LORA_STATE_NAME = "lora_state.msgpack"


@dataclass
class Config:
    # wandb params
    project: str = "toppo-repro"
    group: str = ""
    name: str = ""
    # explicit wandb overrides — applied at the END of __post_init__ so they
    # survive run_experiments.py re-calling __post_init__ after config_args
    # (setting name=/group= directly via config_args is silently clobbered).
    wandb_name: str = ""
    wandb_group: str = ""
    wandb_tags: str = ""  # comma-separated, e.g. "mt50,sac"
    # model params
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    hidden_dim: int = 400
    n_layer: int = 4
    discount: float = 0.99
    tau: float = 5e-3
    n_critic: int = 2
    critic_layernorm: bool = False
    critic_init_layernorm: bool = False
    dropout_rate: float = -1.0  # -1.0 means no dropout
    activation: str = "tanh"  # upstream defaulted to relu; its recipes pass tanh
    rank: int = 16  # 8 for MT10, 16 for MT50
    # training params.  Defaults below = the paper's MT50 protocol, i.e.
    # upstream's published MT50 recipe on Meta-World 3.0.0 with deterministic
    # evaluation (upstream defaulted to n_reset=4, replay_buffer_size=1e6,
    # domain="metaworld", env_name="MT10", eval_interval=1e4, "stc").  The
    # launchers in slurm/ still pass every one of them explicitly.
    batch_size: int = 100
    max_steps: int = int(2e6)  # loop iterations; on MT50 x 50 envs = 100M env steps
    log_interval: int = 1000
    n_reset: int = 6
    threshold: float = 0.65  # 0.8 for MT10, 0.65 for MT50
    replay_buffer_size: int = int(5e5)
    initial_step: int = int(25e3)
    n_train_per_step: int = 1
    # env params
    # metaworld3 = metaworld==3.0.0 / gymnasium, the TOPPO-paper env stack.
    # "metaworld" (upstream's vendored mujoco_py-era copy) does not ship here.
    domain: str = "metaworld3"
    env_name: str = "MT50"
    max_path_length: int = 500
    rf_version: str = "v2"  # metaworld3 reward_function_version ctor kwarg
    # evaluation params
    eval_episodes: int = 1
    eval_interval: int = 20000  # iterations; every 1M env steps on MT50
    # "det" = deterministic policy mean (the TOPPO-paper protocol); "stc" =
    # stochastic sampling (upstream ARS default).
    eval_distribution: str = "det"
    # metaworld3 only: evaluate one episode per benchmark goal (50/task) via
    # set_tasks rounds of 10 parallel copies; False keeps the upstream
    # first-10-goals scheme.
    eval_all_goals: bool = True
    # checkpoint / resume params (see checkpoint_utils.py)
    #   checkpoint_dir: "" = auto  checkpoints/<sanitized group>/<name>_s<seed>
    #     (relative to the CWD, i.e. this directory when launched normally)
    #   checkpoint_interval: iterations between saves; 0 = auto (max_steps//10)
    #   checkpoint_buffer: also persist the replay buffer (multi-GB but required
    #     for a faithful resume — without it the restored agent trains on an
    #     empty buffer and the ARS reward scale loses its history)
    #   resume: "auto" = continue from <checkpoint_dir>/latest when present;
    #           "never" = always start fresh; any other value = explicit path
    #           to a checkpoint dir (the one containing meta.json)
    checkpoint_dir: str = ""
    checkpoint_interval: int = 0
    checkpoint_buffer: bool = True
    resume: str = "auto"
    # general params
    seed: int = 10
    eval_seed: int = 42

    def __post_init__(self):
        if self.domain != "metaworld3":
            # Fail here, before W&B or the env workers start.
            raise ValueError(
                f"domain={self.domain!r} is not supported: this release ships "
                "only the Meta-World 3.0.0 backend (domain=metaworld3); "
                "upstream's vendored Meta-World was removed (see PROVENANCE.md)."
            )
        self.group = f"{self.domain}-{self.env_name}(H={self.max_path_length})"
        reset_interval = math.ceil(self.max_steps / (1000 * (self.n_reset + 1))) * 1000
        if reset_interval % 1e5 == 0:
            r_interval = int(reset_interval / 1e5)
        else:
            r_interval = reset_interval / 1e5
        alg_name = "ARS-LoRA"
        if self.critic_layernorm and self.critic_init_layernorm:
            alg_name += "-LN-ILN"
        elif self.critic_layernorm:
            alg_name += "-LN"

        self.name = f"{alg_name}-NReset{self.n_reset}-Interval{r_interval}e5"
        if self.wandb_group:
            self.group = self.wandb_group
        if self.wandb_name:
            self.name = self.wandb_name


def setup_envs(config: Config):
    envs, env_names, tasks = utils.make_benchmark(
        config.domain, config.env_name, config.seed, config.max_path_length,
        rf_version=config.rf_version,
    )
    env_names = list(env_names)
    env = utils.wrappers.MultiParallelEnvExecutor(envs)
    action_dim = env.action_space.shape[0]
    n_envs = env.n_envs

    eval_rounds = None
    if config.domain == "metaworld3" and config.eval_all_goals:
        # Paper protocol: one eval episode per benchmark goal (50/task),
        # 10 parallel copies x 5 set_tasks rounds per eval.
        import metaworld3_ars
        eval_tasks = {name: list(goals) for name, goals in tasks.items()}
        eval_rounds = metaworld3_ars.build_eval_rounds(eval_tasks, 10)
    else:
        eval_tasks = {
            env_name: tasks_per_env[:10] for env_name, tasks_per_env in tasks.items()
        }
    eval_envs = utils.make_eval_benchmark(
        config.domain,
        config.env_name,
        config.eval_seed,
        eval_tasks,
        config.max_path_length,
        rf_version=config.rf_version,
    )
    eval_env = utils.wrappers.MultiParallelEnvExecutor(eval_envs, 10, eval_tasks)

    return env, eval_env, env_names, action_dim, n_envs, eval_rounds


def _config_fingerprint(config: Config) -> str:
    """Stable hash of the training-relevant config (wandb/checkpoint fields
    excluded) — guards against silently resuming with different physics."""
    import hashlib

    d = asdict(config)
    for k in (
        "project", "group", "name", "wandb_name", "wandb_group", "wandb_tags",
        "checkpoint_dir", "checkpoint_interval", "checkpoint_buffer", "resume",
    ):
        d.pop(k, None)
    blob = str(sorted(d.items())).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _sanitize_path_component(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)


def _lora_state(agent):
    """Serializable pytree of the LoRA-phase models — the counterpart of
    checkpoint_utils._learner_state, which only covers the base models.  The
    frozen base params baked into the LoRA module defs are NOT here: they are
    re-extracted from the restored base actor/critic on resume, so they match
    the original run exactly."""
    return {
        "lora_actor": {
            "step": agent.lora_actor.step,
            "params": agent.lora_actor.params,
            "opt_state": agent.lora_actor.opt_state,
        },
        "lora_critic": {
            "step": agent.lora_critic.step,
            "params": agent.lora_critic.params,
            "opt_state": agent.lora_critic.opt_state,
        },
        "lora_target_critic": {
            "step": agent.lora_target_critic.step,
            "params": agent.lora_target_critic.params,
        },
    }


def main(config: Config):
    # Fork MetaWorld workers before importing JAX/Flax or starting W&B
    # (same order as train_ars.py): forking after JAX creates its thread
    # pools can deadlock in child workers.
    env, eval_env, env_names, action_dim, n_envs, eval_rounds = setup_envs(config)

    from learner_sac_mt_lora import Learner
    import checkpoint_utils as ck

    # Upstream never seeds np.random (replay-buffer sampling uses it), so
    # sampling was irreproducible and run_experiments' sequential-seed loop
    # leaked RNG state across seeds.  Seeding here changes no distribution.
    np.random.seed(config.seed)

    # Don't double the seed suffix when the caller already baked it into the
    # run name (the canonical launchers pass wandb_name=..._s<seed>).
    name_component = _sanitize_path_component(config.name)
    if not name_component.endswith(f"_s{config.seed}"):
        name_component = f"{name_component}_s{config.seed}"
    ckpt_dir = config.checkpoint_dir or os.path.join(
        "checkpoints", _sanitize_path_component(config.group), name_component,
    )
    ckpt_interval = config.checkpoint_interval or max(1, config.max_steps // 10)
    fingerprint = _config_fingerprint(config)

    def _check_meta(meta, path):
        if meta["config_fingerprint"] != fingerprint:
            raise ValueError(
                f"checkpoint at {path} was written with a different "
                f"training config (fingerprint {meta['config_fingerprint']} != "
                f"{fingerprint}). Pass resume=never (or clear the checkpoint "
                "dir) to start fresh, or restore the original config."
            )
        if meta["agent_initialized"] and not meta.get("buffer_saved", False):
            raise ValueError(
                f"checkpoint at {path} has a trained agent but NO replay "
                "buffer (checkpoint_buffer=False). Resuming would silently "
                "train on a near-empty buffer. Re-run with resume=never, or "
                "keep checkpoint_buffer=True for resumable runs."
            )
        rev_then, rev_now = meta.get("git_rev", "unknown"), ck.current_git_rev()
        if rev_then not in ("unknown", rev_now):
            print(
                f"[resume] WARNING: checkpoint written at git rev {rev_then}, "
                f"current rev is {rev_now} — code drift is not fingerprinted; "
                "make sure the training semantics are unchanged."
            )

    resume_candidates = []
    if config.resume == "auto":
        resume_candidates = ck.resumable_candidates(ckpt_dir)
    elif config.resume != "never":
        if not os.path.isfile(os.path.join(config.resume, ck.META_NAME)):
            raise FileNotFoundError(f"resume={config.resume!r} has no {ck.META_NAME}")
        resume_candidates = [config.resume]

    meta = None
    resume_path = None
    if resume_candidates:
        resume_path = resume_candidates[0]
        meta = ck.load_meta(resume_path)
        _check_meta(meta, resume_path)
        print(
            f"[resume] continuing from {resume_path} at iteration "
            f"{meta['iteration']} (wandb run {meta['wandb_run_id']})"
        )

    run_id = meta["wandb_run_id"] if meta is not None else str(uuid.uuid4())
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        tags=[t for t in config.wandb_tags.split(",") if t] or None,
        id=run_id,
        resume="allow",
    )

    # n_envs = TOTAL env slots.  This trainer runs one copy per task, so
    # gpu_k is always 1 and n_tasks == n_envs; the split is kept for
    # structural parity with train_ars.py (learner / replay buffer / alpha /
    # rew_scale are all per-TASK).
    n_tasks = len(env_names)
    gpu_k = n_envs // n_tasks
    kwargs_ = asdict(config)
    kwargs_["hidden_dims"] = tuple([config.hidden_dim for _ in range(config.n_layer)])
    learner_args = inspect.signature(Learner).parameters.keys()

    # Filter config to only include arguments that Learner accepts
    kwargs = {k: v for k, v in kwargs_.items() if k in learner_args}
    agent = None

    replay_buffer = MultiTaskReplayBuffer(
        env.observation_space,
        action_dim,
        # 0/None = "hold the whole run" (gpu_k == 1 keeps the legacy fallback).
        config.replay_buffer_size or config.max_steps * gpu_k,
        n_tasks,
    )

    eval_returns = []
    eval_success = np.zeros(n_tasks)
    observations, dones = env.reset(), False
    rew_scale = np.ones((n_tasks, 1))
    # rew_scale_base: refreshed at every pre-LoRA reset boundary; consumed by
    # the attach branch (a post-attach re-attach reads the LAST pre-LoRA
    # value, so it must survive a resume).  lora_scale: the attach-time
    # rew_scale_base / rew_scale ratio baked into the LoRA module defs —
    # needed to reconstruct them faithfully on resume.
    rew_scale_base = None
    lora_scale = None
    success = np.zeros(n_envs)
    reset_interval = math.ceil(config.max_steps / (1000 * (config.n_reset + 1))) * 1000

    # --- resume: restore buffer, learner (+LoRA), ARS scale, RNG, iteration --
    def _attach_and_restore_lora(agent_, lora_scale_, blob_path):
        """Re-attach the LoRA modules with the checkpointed attach-time scale,
        then overwrite their params/opt_state from the side blob.  The base
        params baked into the LoRA module defs come from the already-restored
        (frozen) base actor/critic, matching the original run exactly."""
        import flax.serialization
        import jax

        saved_rng = agent_.rng  # init_LoRA_modules re-splits it; undo below
        agent_.init_LoRA_modules(
            observations=env.observation_space.sample()[np.newaxis],
            actions=env.action_space.sample()[np.newaxis],
            hidden_dims=kwargs_["hidden_dims"],
            rank=config.rank,
            alpha=1.0,
            scale=lora_scale_,
            actor_lr=kwargs_["actor_lr"],
            critic_lr=kwargs_["critic_lr"],
            n_critic=config.n_critic,
            activation=config.activation,
            critic_layernorm=config.critic_layernorm,
            critic_init_layernorm=config.critic_init_layernorm,
            softplus=True,
        )
        with open(blob_path, "rb") as f:
            blob = f.read()
        template = _lora_state(agent_)
        state = flax.serialization.from_bytes(template, blob)
        dev = jax.tree_util.tree_map(np.asarray, state)
        agent_.lora_actor = agent_.lora_actor.replace(
            step=int(dev["lora_actor"]["step"]),
            params=dev["lora_actor"]["params"],
            opt_state=dev["lora_actor"]["opt_state"],
        )
        agent_.lora_critic = agent_.lora_critic.replace(
            step=int(dev["lora_critic"]["step"]),
            params=dev["lora_critic"]["params"],
            opt_state=dev["lora_critic"]["opt_state"],
        )
        agent_.lora_target_critic = agent_.lora_target_critic.replace(
            step=int(dev["lora_target_critic"]["step"]),
            params=dev["lora_target_critic"]["params"],
        )
        agent_.rng = saved_rng  # checkpointed post-attach rng, not a re-split
        agent_.use_lora = True
        agent_.models.update(
            {
                "alpha": agent_.alpha,
                "lora_actor": agent_.lora_actor,
                "lora_critic": agent_.lora_critic,
            }
        )
        return agent_

    def _restore_from(path, meta_):
        side_path = os.path.join(path, LORA_META_NAME)
        if not os.path.isfile(side_path):
            # _save writes the side files right AFTER checkpoint_utils'
            # atomic dir swap; a kill in that tiny window leaves a checkpoint
            # that checkpoint_utils deems complete but that is torn for the
            # LoRA arm — reject it so the caller falls back to prev.
            raise FileNotFoundError(f"{path} has no {LORA_META_NAME}")
        with open(side_path) as f:
            side = json.load(f)
        ck.load_buffer(path, replay_buffer, meta_)
        restored_agent = None
        if meta_["agent_initialized"]:
            blob = ck.learner_blob(path)
            if blob is None:
                raise FileNotFoundError(
                    f"{path} says agent_initialized but has no learner state"
                )
            restored_agent = Learner(
                observations=env.observation_space.sample()[np.newaxis],
                actions=env.action_space.sample()[np.newaxis],
                n_task=n_tasks,
                **kwargs,
            )
            restored_agent = ck.restore_learner(restored_agent, blob)
            if side.get("lora_attached"):
                lora_blob = os.path.join(path, LORA_STATE_NAME)
                if not os.path.isfile(lora_blob):
                    raise FileNotFoundError(
                        f"{path} says lora_attached but has no {LORA_STATE_NAME}"
                    )
                restored_agent = _attach_and_restore_lora(
                    restored_agent,
                    np.asarray(side["lora_scale"], dtype=np.float32).reshape(
                        n_tasks, 1
                    ),
                    lora_blob,
                )
        restored_scale = None
        if meta_["rew_scale"] is not None:
            # float32 matches the steady-state dtype (rew_scale is derived
            # from float32 reward means once the agent initializes).
            restored_scale = np.asarray(
                meta_["rew_scale"], dtype=np.float32
            ).reshape(n_tasks, 1)
        ck.restore_np_random(meta_)
        r_base = (
            None
            if side.get("rew_scale_base") is None
            else np.asarray(side["rew_scale_base"], dtype=np.float32).reshape(
                n_tasks, 1
            )
        )
        r_lscale = (
            None
            if side.get("lora_scale") is None
            else np.asarray(side["lora_scale"], dtype=np.float32).reshape(n_tasks, 1)
        )
        r_esucc = np.asarray(side["eval_success"], dtype=np.float64)
        return (
            restored_agent,
            restored_scale,
            r_base,
            r_lscale,
            r_esucc,
            int(meta_["iteration"]) + 1,
        )

    start_i = 1
    if meta is not None:
        try:
            r_agent, r_scale, r_base, r_lscale, r_esucc, start_i = _restore_from(
                resume_path, meta
            )
        except Exception as e:
            # latest passed the completeness check but failed to LOAD (e.g. a
            # node crash truncated a file post-rename). Fall back to prev.
            fallback = [c for c in resume_candidates[1:]]
            if not fallback:
                raise
            print(f"[resume] WARNING: loading {resume_path} failed ({e}); "
                  f"falling back to {fallback[0]}")
            meta = ck.load_meta(fallback[0])
            _check_meta(meta, fallback[0])
            r_agent, r_scale, r_base, r_lscale, r_esucc, start_i = _restore_from(
                fallback[0], meta
            )
            # Keep the rejected latest out of the next save's rotation, or that
            # save would prune the prev we just restored from.
            if config.resume == "auto":
                ck.quarantine(resume_path)
        if r_agent is not None:
            agent = r_agent
        if r_scale is not None:
            rew_scale = r_scale
        rew_scale_base = r_base
        lora_scale = r_lscale
        eval_success = r_esucc
        # The env sim state is not checkpointed: episodes restart from reset
        # (same semantics as the PPO GPU path's resume).

    if start_i > config.max_steps:
        # Resubmitting a finished run: don't re-rotate the checkpoint or
        # rewrite multi-GB state for byte-identical content.
        print(f"[resume] run already complete ({config.max_steps} iterations); exiting.")
        env.close()
        eval_env.close()
        wandb.finish()
        return

    def _save(i_now: int) -> None:
        # LoRA-arm side files are staged into the checkpoint's tmp dir and
        # listed in meta.json, so the checkpoint is published (and judged
        # complete) only once they are fully written.
        import flax.serialization
        import jax

        def _write_lora_side(tmp_dir: str):
            attached = agent is not None and agent.use_lora
            side = {
                "lora_attached": bool(attached),
                "rew_scale_base": (
                    None if rew_scale_base is None else np.asarray(rew_scale_base).tolist()
                ),
                "lora_scale": (
                    None if lora_scale is None else np.asarray(lora_scale).tolist()
                ),
                "eval_success": np.asarray(eval_success, dtype=float).tolist(),
            }
            with open(os.path.join(tmp_dir, LORA_META_NAME), "w") as f:
                json.dump(side, f)
            written = [LORA_META_NAME]
            if attached:
                host = jax.device_get(_lora_state(agent))
                with open(os.path.join(tmp_dir, LORA_STATE_NAME), "wb") as f:
                    f.write(flax.serialization.to_bytes(host))
                written.append(LORA_STATE_NAME)
            return written

        path = ck.save_checkpoint(
            ckpt_dir,
            iteration=i_now,
            config_fingerprint=fingerprint,
            wandb_run_id=run_id,
            agent=agent,
            replay_buffer=replay_buffer,
            rew_scale=rew_scale,
            save_buffer=config.checkpoint_buffer,
            extra_writer=_write_lora_side,
        )
        print(f"[checkpoint] saved iteration {i_now} -> {path}")
        if os.environ.get("CHECKPOINT_ASYNC", "0") not in {"1", "true", "True"}:
            ck.offload_checkpoint(path)

    for i in tqdm.tqdm(
        range(start_i, config.max_steps + 1),
        initial=start_i - 1,
        total=config.max_steps,
        smoothing=0.1,
        disable=False,
    ):
        if i > config.initial_step:
            if agent is None:
                mean_rews = np.mean(
                    replay_buffer.rewards[:, : replay_buffer.size], axis=-1
                )
                rew_scale = np.reshape(
                    np.max(mean_rews) / np.maximum(mean_rews, 0.01), (n_tasks, 1)
                )
                utils.log_rew_scale(i, rew_scale, env_names, n_envs_total=n_envs)

                agent = Learner(
                    observations=env.observation_space.sample()[np.newaxis],
                    actions=env.action_space.sample()[np.newaxis],
                    n_task=n_tasks,
                    **kwargs,
                )
            actions = agent.sample_actions(np.array(observations), distribution="stc")
        else:
            actions = [env.action_space.sample() for _ in range(n_envs)]

        next_observations, rewards, dones, infos = env.step(actions)
        masks = []
        dones_float = []
        env_id = 0

        for done, info in zip(dones, infos):
            masks.append(int(not done or "TimeLimit.truncated" in info))
            dones_float.append(float(done))

            if "success" in info:
                success[env_id] += info["success"]
                env_id += 1

        # Task-major slot layout: slot = task_id * gpu_k + copy (gpu_k == 1
        # here, so this is exactly the legacy single insert).
        obs_a = np.asarray(observations)
        next_a = np.asarray(next_observations)
        act_a = np.asarray(actions)
        rew_a = np.asarray(rewards, dtype=np.float32)
        mask_a = np.asarray(masks, dtype=np.float32)
        df_a = np.asarray(dones_float, dtype=np.float32)
        base = np.arange(n_tasks) * gpu_k
        for c in range(gpu_k):
            sl = base + c
            replay_buffer.insert(
                obs_a[sl], act_a[sl], rew_a[sl], mask_a[sl], df_a[sl], next_a[sl]
            )

        observations = next_observations

        if all(dones):
            observations, dones = env.reset(), False
            if i % config.log_interval:
                utils.log_episode_info(
                    i, infos, env_names, config.env_name, success,
                    n_envs_total=n_envs,
                )
            success = np.zeros(n_envs)

        if i * gpu_k > config.batch_size and i > config.initial_step:
            for _ in range(config.n_train_per_step):
                batch = replay_buffer.sample(config.batch_size, rew_scale=rew_scale)
                update_info = agent.update(batch)

            if i % config.log_interval == 0:
                utils.log_update_metrics(
                    i, update_info, env_names, n_envs_total=n_envs
                )

        if i % config.eval_interval == 0 and i > config.initial_step:
            eval_returns_, eval_success = utils.run_eval(
                i,
                agent,
                eval_env,
                config.eval_episodes,
                # tasks, not slots: evaluate_n_envs keeps one stat bucket per
                # task and derives copies-per-task from len(obs)//n_envs
                n_tasks,
                env_names,
                config.env_name,
                n_envs_total=n_envs,
                distribution=config.eval_distribution,
                tasks=eval_rounds,
            )
            eval_returns.append((i, *eval_returns_))

        # i < max_steps: no reset/attach at the final iteration (MT10's
        # 400k interval divides 2M); the final evaluation precedes it anyway.
        if i % reset_interval == 0 and i > config.batch_size and i < config.max_steps:
            if not agent.use_lora:
                rew_scale_base = rew_scale.copy()
            mean_rews = np.mean(replay_buffer.rewards[:, : replay_buffer.size], axis=-1)
            rew_scale = np.reshape(
                np.max(mean_rews) / np.maximum(mean_rews, 0.01), (n_tasks, 1)
            )
            utils.log_rew_scale(i, rew_scale, env_names, n_envs_total=n_envs)
            if (
                np.mean(eval_success) < config.threshold and not agent.use_lora
            ) or i <= config.max_steps * 0.75:
                agent = Learner(
                    observations=env.observation_space.sample()[np.newaxis],
                    actions=env.action_space.sample()[np.newaxis],
                    n_task=n_tasks,
                    **kwargs,
                )
            else:
                lora_scale = rew_scale_base / rew_scale
                agent.init_LoRA_modules(
                    observations=env.observation_space.sample()[np.newaxis],
                    actions=env.action_space.sample()[np.newaxis],
                    hidden_dims=kwargs_["hidden_dims"],
                    rank=config.rank,
                    alpha=1.0,
                    scale=lora_scale,
                    actor_lr=kwargs_["actor_lr"],
                    critic_lr=kwargs_["critic_lr"],
                    n_critic=config.n_critic,
                    activation=config.activation,
                    critic_layernorm=config.critic_layernorm,
                    critic_init_layernorm=config.critic_init_layernorm,
                    softplus=True,
                )
                agent.use_lora = True

            for _ in range(int(1e3)):
                batch = replay_buffer.sample(config.batch_size, rew_scale=rew_scale)
                update_info = agent.update(batch)

        # Periodic checkpoint — last block of the iteration so the saved
        # state reflects a fully completed iteration (incl. any ARS reset
        # or LoRA attach).
        if i % ckpt_interval == 0 and i < config.max_steps:
            _save(i)

    _save(config.max_steps)
    env.close()
    eval_env.close()
    wandb.finish()


if __name__ == "__main__":
    config = pyrallis.parse(Config)
    main(config)
