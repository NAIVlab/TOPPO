import os
import inspect

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
    wandb_tags: str = ""  # comma-separated, e.g. "mt50,baseline"
    # model params
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_dim: int = 400
    n_layer: int = 4
    discount: float = 0.99
    tau: float = 5e-3
    n_critic: int = 2
    critic_layernorm: bool = False
    critic_init_layernorm: bool = False
    dropout_rate: float = -1.0  # -1.0 means no dropout
    activation: str = "tanh"
    multi_head: bool = False
    use_pcgrad: bool = False
    # critic balancing (TOPPO critic stack in SAC) — forwarded to the Learner.
    # The full CB arm = critic_method=fairgrad + use_popart=True +
    # critic_layernorm=True (LN-c); actor_method=pcgrad adds the paper's
    # actor-side PCGrad.
    actor_method: str = "default"   # default | pcgrad
    critic_method: str = "default"  # default | pcgrad | fairgrad (FG-c, alpha=1)
    fairgrad_critic_alpha: float = 1.0
    use_popart: bool = False
    popart_method: str = "ema"      # ema (Hessel beta=3e-4) | welford (paper)
    popart_ema_decay: float = 3e-4
    popart_min_std: float = 1e-2
    # training params.  Defaults below = the paper's MT50 protocol, i.e.
    # upstream's published MT50 recipe on Meta-World 3.0.0 with deterministic
    # evaluation (upstream defaulted to n_reset=4, replay_buffer_size=1e6,
    # domain="metaworld", env_name="MT10", eval_interval=1e4, "stc").  The
    # launchers in slurm/ still pass every one of them explicitly.
    use_ars: bool = True
    batch_size: int = 100
    max_steps: int = int(2e6)  # loop iterations; on MT50 x 50 envs = 100M env steps
    log_interval: int = 1000
    n_reset: int = 6
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
    eval_copies_per_task: int = 10
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

        if self.use_ars:
            alg_name = "ARS(SAC-MT-MH)" if self.multi_head else "ARS"
        else:
            alg_name = "SAC-MT-MH" if self.multi_head else "SAC-MT"
        if self.use_pcgrad:
            alg_name += "-PCGrad"
        if self.critic_layernorm and self.critic_init_layernorm:
            alg_name += "-LN-ILN"
        elif self.critic_layernorm:
            alg_name += "-LN"
        if self.critic_method == "fairgrad":
            alg_name += "-FGc"
        elif self.critic_method == "pcgrad" and not self.use_pcgrad:
            alg_name += "-PCGc"
        if self.actor_method == "pcgrad" and not self.use_pcgrad:
            alg_name += "-PCGa"
        if self.use_popart:
            alg_name += "-PopArt"

        self.name = (
            f"{alg_name}-NReset{self.n_reset}-Interval{r_interval}e5"
            if self.use_ars
            else f"{alg_name}"
        )
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
        eval_rounds = metaworld3_ars.build_eval_rounds(eval_tasks, config.eval_copies_per_task)
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
    eval_env = utils.wrappers.MultiParallelEnvExecutor(eval_envs, config.eval_copies_per_task, eval_tasks)

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


def main(config: Config):
    # Fork MetaWorld workers before importing JAX/Flax or starting W&B.
    # Forking after JAX creates its thread pools can deadlock in child workers.
    env, eval_env, env_names, action_dim, n_envs, eval_rounds = setup_envs(config)

    from learner_sac_mt import Learner
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

    # n_envs = TOTAL env slots; an executor with gpu_k parallel copies per
    # task lays them out task-major.  The learner / replay buffer / alpha /
    # rew_scale all stay per-TASK (n_task = n_tasks).  The shipped Meta-World
    # executor runs one copy per task (gpu_k == 1): upstream's path exactly.
    n_tasks = len(env_names)
    gpu_k = n_envs // n_tasks
    if gpu_k > 1 and config.multi_head:
        raise ValueError(
            "More than one parallel copy per task (gpu_k > 1) with "
            "multi_head=True is unsupported: sample_actions receives flat "
            "(n_tasks*K, obs) collection batches and the multi-head policy "
            "routes heads by axis-0 index, so task-major slots would silently "
            "use the wrong heads. Use multi_head=False, or one copy per task."
        )
    kwargs_ = asdict(config)
    kwargs_["hidden_dims"] = tuple([config.hidden_dim for _ in range(config.n_layer)])
    kwargs_["multi_head"] = config.multi_head
    learner_args = inspect.signature(Learner).parameters.keys()
    kwargs = {k: v for k, v in kwargs_.items() if k in learner_args}
    agent = None

    replay_buffer = MultiTaskReplayBuffer(
        env.observation_space,
        action_dim,
        # 0/None = "hold the whole run": each iteration inserts gpu_k columns
        # per task, so the no-wrap capacity is max_steps * gpu_k (gpu_k == 1
        # on the CPU path, reducing to the legacy fallback exactly).
        config.replay_buffer_size or config.max_steps * gpu_k,
        n_tasks,
    )

    observations, dones = env.reset(), False
    rew_scale = np.ones((n_tasks, 1)) if config.use_ars else None
    success = np.zeros(n_envs)
    reset_interval = (
        math.ceil(config.max_steps / (1000 * (config.n_reset + 1))) * 1000
        if config.use_ars
        else config.max_steps + 1
    )

    # --- resume: restore buffer, learner, ARS scale, RNG, iteration ---------
    def _restore_from(path, meta_):
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
        restored_scale = None
        if config.use_ars and meta_["rew_scale"] is not None:
            # float32 matches the steady-state dtype (rew_scale is derived
            # from float32 reward means once the agent initializes).
            restored_scale = np.asarray(
                meta_["rew_scale"], dtype=np.float32
            ).reshape(n_tasks, 1)
        ck.restore_np_random(meta_)
        return restored_agent, restored_scale, int(meta_["iteration"]) + 1

    start_i = 1
    if meta is not None:
        try:
            r_agent, r_scale, start_i = _restore_from(resume_path, meta)
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
            r_agent, r_scale, start_i = _restore_from(fallback[0], meta)
            # Keep the rejected latest out of the next save's rotation, or that
            # save would prune the prev we just restored from.
            if config.resume == "auto":
                ck.quarantine(resume_path)
        if r_agent is not None:
            agent = r_agent
        if r_scale is not None:
            rew_scale = r_scale
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
        path = ck.save_checkpoint(
            ckpt_dir,
            iteration=i_now,
            config_fingerprint=fingerprint,
            wandb_run_id=run_id,
            agent=agent,
            replay_buffer=replay_buffer,
            rew_scale=rew_scale if config.use_ars else None,
            save_buffer=config.checkpoint_buffer,
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
                if config.use_ars:
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

        # Task-major slot layout: slot = task_id * gpu_k + copy.  One insert
        # per copy keeps the buffer's per-task rows (gpu_k == 1 -> identical
        # to the legacy single insert).
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

        # Gate on collected samples PER TASK (i * gpu_k), not iterations:
        # with K env copies per task the buffer reaches batch_size after
        # batch_size/K iterations, and initial_step keeps its documented
        # learning-starts meaning.  gpu_k == 1 reproduces the legacy gate.
        if i * gpu_k > config.batch_size and i > config.initial_step:
            sample_kwargs = {"rew_scale": rew_scale} if config.use_ars else {}
            for _ in range(config.n_train_per_step):
                batch = replay_buffer.sample(config.batch_size, **sample_kwargs)
                update_info = agent.update(batch)

            if i % config.log_interval == 0:
                utils.log_update_metrics(
                    i, update_info, env_names, n_envs_total=n_envs
                )

        if i % config.eval_interval == 0 and i > config.initial_step:
            utils.run_eval(
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

        # i < max_steps: no reset at the final iteration (MT10's 400k interval
        # divides 2M); the final evaluation precedes it anyway.
        if (
            config.use_ars
            and i % reset_interval == 0
            and i > config.batch_size
            and i < config.max_steps
        ):
            mean_rews = np.mean(replay_buffer.rewards[:, : replay_buffer.size], axis=-1)
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
            for _ in range(int(1e3)):
                batch = replay_buffer.sample(config.batch_size, rew_scale=rew_scale)
                update_info = agent.update(batch)

        # Periodic checkpoint — last block of the iteration so the saved
        # state reflects a fully completed iteration (incl. any ARS reset).
        if i % ckpt_interval == 0 and i < config.max_steps:
            _save(i)

    _save(config.max_steps)
    env.close()
    eval_env.close()
    wandb.finish()


if __name__ == "__main__":
    config = pyrallis.parse(Config)
    main(config)
