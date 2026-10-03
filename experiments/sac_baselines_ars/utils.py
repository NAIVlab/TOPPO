import sys
from pathlib import Path

import numpy as np
import wandb
import gymnasium as gym  # type hints only; the old-gym paths import lazily

# Prefer the local metaworld package in this repo over any installed version.
_REPO_ROOT = Path(__file__).resolve().parent
_LOCAL_METAWORLD = _REPO_ROOT / "metaworld"
if _LOCAL_METAWORLD.exists():
    sys.path.insert(0, str(_REPO_ROOT))

# NOTE: metaworld is imported lazily inside the metaworld branch of
# make_benchmark/make_eval_benchmark so an unused backend does not pull
# in mujoco_py (which requires MUJOCO_PY_MUJOCO_PATH / mujoco210 on the path).

import wrappers


def _env_step(step, env_names, n_envs_total=None):
    """mtrl-aligned wandb x-axis: TOTAL env interactions across all envs.

    One training-loop iteration steps every env slot once, so
    env_step = step * n_envs_total (defaults to len(env_names) — one env per
    task, the shipped executor's case).  This matches the PPO side's x-axis
    (``env_step`` = total interactions across all N parallel envs), so SAC
    and PPO curves overlay directly in wandb.  Passed as the
    explicit ``wandb.log(step=...)`` (all call sites are monotonic in the
    loop iteration); the legacy "time step" key (= per-env step) is kept for
    old panels.
    """
    return int(step) * int(n_envs_total or len(env_names))


def log_rew_scale(step, rew_scale, env_names, n_envs_total=None):
    es = _env_step(step, env_names, n_envs_total)
    wandb.log(
        {
            "time step": step,
            "env_step": es,
            **{
                f"{env_name}/training/rew_scale": np.mean(rew_scale[env_id])
                for env_id, env_name in enumerate(env_names)
            },
        },
        step=es,
    )


def log_episode_info(step, infos, env_names, env_name, success, n_envs_total=None):
    es = _env_step(step, env_names, n_envs_total)
    wandb.log(
        {
            "time step": step,
            "env_step": es,
            f"{env_name}/training/average_successs": np.mean(success > 0),
            # mtrl-aligned TRAINING-rollout keys (success-once per training
            # episode; PPO's rollout scope) — eval keys live in run_eval.
            "rollout/mean_win_rate": float(np.mean(success > 0)),
            **(
                {
                    f"win_rate/{name}": float(success[i] > 0)
                    for i, name in enumerate(env_names)
                }
                if len(infos) == len(env_names)
                else {}
            ),
        },
        step=es,
    )
    k = len(infos) // len(env_names)
    if k > 1:
        # Executor with k copies per task (task-major slots; unused by the
        # shipped Meta-World executor, which runs one copy per task).
        # Aggregate each task's k copies (episode stats -> mean over copies
        # that finished; success -> fraction of copies with success-once).
        for tid, name in enumerate(env_names):
            block = infos[tid * k:(tid + 1) * k]
            eps = [inf["episode"] for inf in block if "episode" in inf]
            if eps:
                wandb.log(
                    {
                        "time step": step,
                        **{
                            f"{name}/training/average_{kk}s": float(
                                np.mean([ep[kk] for ep in eps])
                            )
                            for kk in eps[0]
                            if kk not in ("length", "duration")
                        },
                    },
                    step=es,
                )
            if any("success" in inf for inf in block):
                wandb.log(
                    {
                        "time step": step,
                        f"{name}/training/average_success": float(
                            np.mean(np.asarray(success[tid * k:(tid + 1) * k]) > 0)
                        ),
                    },
                    step=es,
                )
        return
    for env_id, info in enumerate(infos):
        for key, val in info.items():
            if key == "episode":
                val.pop("length")
                val.pop("duration")
                wandb.log(
                    {
                        "time step": step,
                        **{
                            f"{env_names[env_id]}/training/average_{k}s": v
                            for k, v in val.items()
                        },
                    },
                    step=es,
                )
            elif key == "success":
                wandb.log(
                    {
                        "time step": step,
                        f"{env_names[env_id]}/training/average_success": int(
                            success[env_id] > 0
                        ),
                    },
                    step=es,
                )


def log_update_metrics(step, update_info, env_names, n_envs_total=None):
    es = _env_step(step, env_names, n_envs_total)
    for k, v in update_info.items():
        if v.ndim == 0:
            wandb.log({"time step": step, "env_step": es, f"training/{k}": v}, step=es)
        else:
            wandb.log(
                {
                    f"time step": step,
                    **{
                        f"{env_name}/training/{k}": vv
                        for env_name, vv in zip(env_names, v)
                    },
                },
                step=es,
            )


def run_eval(step, agent, eval_env, eval_episodes, n_envs, env_names, env_name,
             n_envs_total=None, distribution="stc", tasks=None):
    from evaluation import evaluate_n_envs, evaluates
    # tasks: optional list of set_tasks rounds (metaworld3 50-goal protocol —
    # see metaworld3_ars.build_eval_rounds); when given, evaluate_n_envs runs
    # one episode per round and eval_episodes is ignored.
    eval_stats = evaluates(
        eval_f=evaluate_n_envs,
        agent=agent,
        env=eval_env,
        num_episodes=eval_episodes,
        n_envs=n_envs,
        distribution=distribution,
        tasks=tasks,
    )
    eval_returns = [eval_stat["return"] for eval_stat in eval_stats]
    if "success" in eval_stats[0].keys():
        eval_success = [eval_stat["success"] for eval_stat in eval_stats]
    else:
        eval_success = [0 for _ in range(n_envs)]
    es = _env_step(step, env_names, n_envs_total)
    wandb.log(
        {
            "time step": step,
            "env_step": es,
            # legacy key (kept so old panels/queries keep working)
            f"{env_name}/evaluation/average_successs": np.mean(eval_success),
            # mtrl-aligned EVAL keys — exactly the multi-task PPO test scope
            # (logger.py: test/mean_win_rate + test_win_rate/{task}), so SAC
            # and PPO MT50 success rates are the same metric, no conversion:
            # success-once over one episode per benchmark goal, mean over
            # goals then tasks.
            "test/mean_win_rate": float(np.mean(eval_success)),
            # NOTE: deliberately NOT named worst/success — mtrl's worst/success
            # is a TRAINING-side min with carry-forward; this is the eval-side
            # min (deterministic, 50 goals).  Distinct key avoids overlaying
            # different distributions in one panel.
            "test/worst_win_rate": float(np.min(eval_success)),
            **{
                f"test_win_rate/{env_name_item}": float(eval_stat["success"])
                for env_name_item, eval_stat in zip(env_names, eval_stats)
                if "success" in eval_stat
            },
        },
        step=es,
    )
    for env_name_item, eval_stat in zip(env_names, eval_stats):
        wandb.log(
            {
                "time step": step,
                **{
                    f"{env_name_item}/evaluation/average_{k}s": v
                    for k, v in eval_stat.items()
                    if "length" not in k
                },
            },
            step=es,
        )
    return eval_returns, eval_success


def make_benchmark(
    domain: str,
    benchmark_name: str,
    seed: int,
    max_path_length: int = 500,
    rand_g: bool = True,
    reward_mode: str = "normalized_dense",
    obs_dim: int = 100,
    rf_version: str = "v2",
) -> gym.Env:
    if domain == "metaworld3":
        import metaworld3_ars
        return metaworld3_ars.get_mw3_envs(
            benchmark_name, seed, max_path_length,
            reward_function_version=rf_version)
    import metaworld
    try:
        benchmark_class = getattr(metaworld, benchmark_name)
        benchmark = benchmark_class(seed=seed)
    except TypeError:
        benchmark = benchmark_class()
    except AttributeError:
        raise NotImplementedError(f"{benchmark_name} is not implemented in {domain}.")

    envs = []
    tasks_per_env = {}
    tasks = benchmark.train_tasks
    env_names = list(benchmark.train_classes.keys())

    for env_id, env_cls in enumerate(benchmark.train_classes.values()):
        env = env_cls()
        env.set_task(tasks[env_id * 50])
        env._freeze_rand_vec = not rand_g

        env = wrappers.EpisodeMonitor(env)

        env.seed(seed + env_id + 1)
        env.action_space.seed(seed + env_id + 1)
        env.observation_space.seed(seed + env_id + 1)

        env.max_path_length = max_path_length

        envs.append(env)

        tasks_per_env[env_names[env_id]] = tasks[env_id * 50 : (env_id + 1) * 50]

    return envs, env_names, tasks_per_env


def make_eval_benchmark(
    domain: str, benchmark_name: str, seed: int, tasks, max_path_length: int = 500,
    reward_mode: str = "normalized_dense", obs_dim: int = 100,
    rf_version: str = "v2",
) -> gym.Env:
    if domain == "metaworld3":
        import metaworld3_ars
        return metaworld3_ars.get_mw3_eval_envs(
            benchmark_name, seed, tasks, max_path_length,
            reward_function_version=rf_version)
    import metaworld
    try:
        benchmark_class = getattr(metaworld, benchmark_name)
        benchmark = benchmark_class(seed=seed)
    except TypeError:
        benchmark = benchmark_class()
    except AttributeError:
        raise NotImplementedError(f"{benchmark_name} is not implemented in {domain}.")

    envs = []
    env_names = list(benchmark.train_classes.keys())

    for env_id, env_cls in enumerate(benchmark.train_classes.values()):
        env = env_cls()
        env.set_task(tasks[env_names[env_id]][0])

        env = wrappers.EpisodeMonitor(env)

        env.seed(seed + env_id + 1)
        env.action_space.seed(seed + env_id + 1)
        env.observation_space.seed(seed + env_id + 1)

        env.max_path_length = max_path_length

        envs.append(env)

    return envs
