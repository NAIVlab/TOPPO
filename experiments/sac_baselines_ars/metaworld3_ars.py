"""MetaWorld 3.0.0 (gymnasium) adapter for the ARS/SAC training loop.

Runs the ARS-family trainers on the SAME environment stack as the TOPPO
paper (metaworld==3.0.0, new mujoco bindings, V2 reward, H=500) instead of
the vendored mujoco_py-era MetaWorld.  Follows the sibling backend
lazy-proxy recipe: the JAX parent process never imports metaworld/mujoco —
each executor worker builds its env on first use.

Fidelity to the paper's training env (mtrl metaworld_wrapper._make_single_env):
  - benchmark        : metaworld.MT10/MT50(seed=<train seed>) — goals fixed
                       by the benchmark seed, 50 per task.
  - training goals   : metaworld.wrappers.RandomTaskSelectWrapper over the 50
                       benchmark goals, resampled on every reset (the paper's
                       ``task_select: random``) — NOT the upstream-ARS
                       unlimited ``_freeze_rand_vec=False`` goal space.
  - reward           : ``reward_function_version`` ctor kwarg ("v2" default).
  - horizon          : fixed 500, truncation-only, bootstrapped (mask=1) —
                       enforced by the monitor below, mirroring both the
                       vendored EpisodeMonitor and the paper's TimeLimit.
  - one-hot task id  : appended by ``MultiParallelEnvExecutor`` (unchanged).
  - eval             : deterministic policy, one episode per benchmark goal
                       (50/task) — driven from train_ars via ``set_task``
                       rounds; ``MW3LazyEnv.set_task`` accepts the pickled
                       Task objects.

The proxy speaks the old-gym 4-tuple protocol the executor expects:
``(obs, reward, done, info)`` with ``info["TimeLimit.truncated"]`` on
timeout dones and per-step ``info["success"]``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium.spaces import Box

MW3_OBS_DIM = 39
MW3_ACT_DIM = 4

# Per-process benchmark cache: eval workers hold 10 env copies each, and
# every copy would otherwise construct its own metaworld.MT50() (2500 task
# pickles).  Keyed by (benchmark, seed); safe because Benchmark objects are
# read-only after construction.
_BENCH_CACHE = {}


def _get_benchmark(benchmark_name: str, seed: int):
    import metaworld

    key = (benchmark_name, seed)
    if key not in _BENCH_CACHE:
        _BENCH_CACHE[key] = getattr(metaworld, benchmark_name)(seed=seed)
    return _BENCH_CACHE[key]


class MW3LazyEnv:
    """Picklable lazy proxy around one MetaWorld-3 task env.

    ``__init__`` stores config + synthetic spaces only; the real env is
    built inside the executor worker on first ``reset``/``set_task`` use.
    """

    def __init__(
        self,
        task_name: str,
        benchmark_name: str,
        benchmark_seed: int,
        seed: int,
        max_path_length: int = 500,
        reward_function_version: str = "v2",
        randomize_goals: bool = True,
        task_cls=None,
        goals=None,
    ):
        self.task_name = task_name
        self.benchmark_name = benchmark_name
        self.benchmark_seed = benchmark_seed
        self.seed_value = seed
        self.max_path_length = max_path_length
        self.reward_function_version = reward_function_version
        self.randomize_goals = randomize_goals
        self.task_cls = task_cls
        self.goals = list(goals) if goals is not None else None

        self.observation_space = Box(
            -np.inf, np.inf, (MW3_OBS_DIM,), dtype=np.float64
        )
        self.action_space = Box(-1.0, 1.0, (MW3_ACT_DIM,), dtype=np.float32)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

        self._env = None
        self._pending_task = None
        self._needs_seed = True
        # episode-monitor state (mirrors wrappers.EpisodeMonitor)
        self._reward_sum = 0.0
        self._reward_sq = 0.0
        self._episode_length = 0
        self._start_time = time.time()
        self.total_timesteps = 0

    # -- pickling: never ship a live mujoco env across processes ------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_needs_seed"] = True
        return state

    def _build(self):
        if self.task_cls is None or self.goals is None:
            bench = _get_benchmark(self.benchmark_name, self.benchmark_seed)
            task_cls = bench.train_classes[self.task_name]
            goals = [t for t in bench.train_tasks if t.env_name == self.task_name]
        else:
            task_cls = self.task_cls
            goals = self.goals
        assert goals, f"no tasks for {self.task_name} in {self.benchmark_name}"

        env = task_cls(
            render_mode=None,
            reward_function_version=self.reward_function_version,
        )
        env.set_task(goals[0])
        # SawyerXYZEnv.reset(seed=...) IGNORES the seed by design ("use
        # seed() instead") — seed explicitly so goal selection (the
        # RandomTaskSelectWrapper draws from env.np_random) and reset noise
        # are reproducible per (run seed, task).
        env.seed(self.seed_value)
        if self.randomize_goals:
            from metaworld.wrappers import RandomTaskSelectWrapper

            env = RandomTaskSelectWrapper(
                env, tasks=goals, sample_tasks_on_reset=True
            )
        self._env = env
        self._goals = goals

    def _ensure_env(self):
        if self._env is None:
            self._build()
            if self._pending_task is not None:
                self._apply_task(self._pending_task)
                self._pending_task = None

    def _apply_task(self, task):
        base = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
        base.set_task(task)

    # -- old-gym API expected by MultiParallelEnvExecutor -------------------
    def set_task(self, task):
        """Eval goal assignment (pickled metaworld Task objects)."""
        if task is None:
            return
        if self._env is None:
            self._pending_task = task
            return
        self._apply_task(task)

    def seed(self, seed: int):
        self.seed_value = seed
        self._needs_seed = True
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def reset(self) -> np.ndarray:
        self._ensure_env()
        if self._needs_seed:
            # RNG seeding happens via env.seed() in _build (reset(seed=) is a
            # documented no-op in metaworld 3.0); flag kept for re-seed after
            # unpickling into a worker.
            self._needs_seed = False
        obs, _ = self._env.reset()
        self._reward_sum = 0.0
        self._reward_sq = 0.0
        self._episode_length = 0
        self._start_time = time.time()
        return np.asarray(obs, dtype=np.float64)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self._ensure_env()
        action = np.asarray(action, dtype=np.float32).reshape(
            self.action_space.shape
        )
        obs, reward, terminated, truncated, info = self._env.step(action)
        reward = float(reward)

        self._reward_sum += reward
        self._reward_sq += reward**2
        self._episode_length += 1
        self.total_timesteps += 1

        out_info: Dict[str, Any] = {
            "success": float(info.get("success", 0.0)),
            "total": {"timesteps": self.total_timesteps},
        }

        # MetaWorld MT envs never terminate; the fixed horizon is ours to
        # enforce (truncation-only, bootstrapped by the train loop's mask
        # logic via TimeLimit.truncated — same as the vendored path).
        done = bool(terminated or truncated)
        if self._episode_length >= self.max_path_length and not done:
            done = True
            out_info["TimeLimit.truncated"] = True
        elif truncated and not terminated:
            out_info["TimeLimit.truncated"] = True

        if done:
            ep_len = max(self._episode_length, 1)
            rew_mean = self._reward_sum / ep_len
            rew_var = max(self._reward_sq / ep_len - rew_mean**2, 0.0)
            out_info["episode"] = {
                "return": self._reward_sum,
                "length": self._episode_length,
                "rew_mean": rew_mean,
                "rew_std": rew_var**0.5,
                "duration": time.time() - self._start_time,
            }

        return np.asarray(obs, dtype=np.float64), reward, done, out_info

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


def get_mw3_envs(
    benchmark_name: str,
    seed: int,
    max_path_length: int = 500,
    reward_function_version: str = "v2",
):
    """Training envs — mirrors utils.make_benchmark's return contract:
    ``(envs, env_names, tasks_per_env)`` with tasks_per_env carrying the 50
    benchmark Task objects per task (consumed by eval-round construction)."""
    bench = _get_benchmark(benchmark_name, seed)
    env_names = sorted(bench.train_classes.keys())

    tasks_per_env: Dict[str, List] = {name: [] for name in env_names}
    for t in bench.train_tasks:
        tasks_per_env[t.env_name].append(t)

    envs = []
    for env_id, name in enumerate(env_names):
        envs.append(
            MW3LazyEnv(
                task_name=name,
                benchmark_name=benchmark_name,
                benchmark_seed=seed,
                seed=seed + env_id + 1,
                max_path_length=max_path_length,
                reward_function_version=reward_function_version,
                randomize_goals=True,
                task_cls=bench.train_classes[name],
                goals=tasks_per_env[name],
            )
        )
    return envs, env_names, tasks_per_env


def get_mw3_eval_envs(
    benchmark_name: str,
    seed: int,
    tasks: Dict[str, List],
    max_path_length: int = 500,
    reward_function_version: str = "v2",
):
    """Eval envs — one lazy proxy per task, goals frozen (assigned via
    executor ``set_tasks`` rounds).  ``tasks`` keys give the task order."""
    bench = _get_benchmark(benchmark_name, seed)
    envs = []
    for env_id, name in enumerate(tasks.keys()):
        envs.append(
            MW3LazyEnv(
                task_name=name,
                benchmark_name=benchmark_name,
                # goals come from the TRAIN benchmark seed via set_tasks;
                # this seed only controls reset noise in the eval envs.
                benchmark_seed=seed,
                seed=seed + env_id + 1,
                max_path_length=max_path_length,
                reward_function_version=reward_function_version,
                randomize_goals=False,
                task_cls=bench.train_classes[name],
                goals=tasks[name],
            )
        )
    return envs


def build_eval_rounds(
    tasks_per_env: Dict[str, List], copies_per_task: int
) -> List[List[List]]:
    """Partition each task's goal list into ``set_tasks`` rounds.

    Returns ``rounds`` where ``rounds[r][worker]`` is the list of
    ``copies_per_task`` Task objects worker ``worker`` evaluates in round
    ``r`` — evaluate_n_envs runs one episode per round with
    ``env.set_tasks(rounds[r])``.  50 goals / 10 copies -> 5 rounds, one
    episode per goal (the paper's protocol).
    """
    names = list(tasks_per_env.keys())
    n_goals = min(len(tasks_per_env[n]) for n in names)
    n_rounds = n_goals // copies_per_task
    assert n_rounds * copies_per_task == n_goals, (
        f"goals ({n_goals}) must be divisible by copies ({copies_per_task})"
    )
    rounds = []
    for r in range(n_rounds):
        lo, hi = r * copies_per_task, (r + 1) * copies_per_task
        rounds.append([tasks_per_env[name][lo:hi] for name in names])
    return rounds
