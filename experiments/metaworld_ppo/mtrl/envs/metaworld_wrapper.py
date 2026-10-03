"""MetaWorld MT10 / MT25 / MT50 vectorized environments + task wrappers."""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from mtrl.utils.config import Config


class TaskIDWrapper(gym.Wrapper):
    """Append a one-hot task id to the observation."""

    def __init__(
        self,
        env: gym.Env,
        task_id: int,
        num_tasks: int,
        task_name: str,
    ) -> None:
        super().__init__(env)
        self.task_id = task_id
        self.num_tasks = num_tasks
        self.task_name = task_name

        low = self.observation_space.low.astype(np.float32)
        high = self.observation_space.high.astype(np.float32)
        one_hot_low = np.zeros(num_tasks, dtype=np.float32)
        one_hot_high = np.ones(num_tasks, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low, one_hot_low]),
            high=np.concatenate([high, one_hot_high]),
            dtype=np.float32,
        )

        self._one_hot = np.zeros(num_tasks, dtype=np.float32)
        self._one_hot[task_id] = 1.0

    def _add_task_id(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, self._one_hot])

    @property
    def reward_norm_state(self):
        """Expose the baselines_return normalizer state through the outermost
        wrapper: gymnasium ≥1.0 removed Wrapper attribute delegation, so
        tianshou's vec-env ``getattr`` (which queries the outermost wrapper)
        would otherwise never see the dict stored on ``env.unwrapped``.
        None when reward normalization is off."""
        return getattr(self.unwrapped, "reward_norm_state", None)

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        info["task_id"] = self.task_id
        info["task_name"] = self.task_name
        return self._add_task_id(obs), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["task_id"] = self.task_id
        info["task_name"] = self.task_name

        info["win"] = float(info["success"]) if "success" in info else 0.0
        return self._add_task_id(obs), reward, terminated, truncated, info


class RewardNormalizationWrapper(gym.Wrapper):
    """Reward normalization with ``"gymnasium"`` (running mean/std of the
    reward), ``"exponential"`` (EMA), or ``"baselines_return"`` (openai/
    baselines ``VecNormalize``: divide the raw reward by the running std of
    the discounted-return accumulator, no mean subtraction, clip to
    ``±clip``; see openai/baselines#538)."""

    def __init__(
        self,
        env: gym.Env,
        method: str = "gymnasium",
        gamma: float = 0.99,
        clip: float = 10.0,
    ) -> None:
        super().__init__(env)
        self.method = method
        self.gamma = gamma
        self.clip = clip

        self._running_mean = 0.0
        self._running_var = 1.0
        self._count = 1e-8

        self._ema_reward = 0.0
        self._ema_var = 1.0

        # baselines_return state.  The accumulator ret_t = γ·ret_{t−1} + r_t
        # resets at every episode boundary; the RunningMeanStd over ret values
        # (mean/var/count, count seeded with 1e-4) persists for the whole run.
        #
        # The state dict lives on ``env.unwrapped`` rather than on this
        # wrapper: tianshou's vec-env ``getattr`` walks the wrapper chain and
        # its ``setattr`` targets ``env.unwrapped``, so keeping a single dict
        # there lets train.py snapshot the statistics into checkpoints
        # (get_env_attr) and push them back after a preemption resume
        # (set_env_attr) — otherwise every requeue would silently reset the
        # reward scale mid-run.  Always re-read through the property; a
        # restore REPLACES the dict object.
        if method == "baselines_return":
            self.env.unwrapped.reward_norm_state = {
                "ret": 0.0, "mean": 0.0, "var": 1.0, "count": 1e-4,
            }

    @property
    def _brn_state(self) -> Dict[str, float]:
        return self.env.unwrapped.reward_norm_state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["raw_reward"] = reward
        reward = self._normalize(float(reward), terminated or truncated)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.method == "baselines_return":
            self._brn_state["ret"] = 0.0
        return self.env.reset(**kwargs)

    def _normalize(self, reward: float, done: bool) -> float:
        if self.method == "gymnasium":
            self._count += 1
            delta = reward - self._running_mean
            self._running_mean += delta / self._count
            delta2 = reward - self._running_mean
            self._running_var += (delta * delta2 - self._running_var) / self._count
            return reward / (np.sqrt(self._running_var) + 1e-8)
        elif self.method == "exponential":
            self._ema_reward = self.gamma * self._ema_reward + (1 - self.gamma) * reward
            self._ema_var = self.gamma * self._ema_var + (1 - self.gamma) * (reward - self._ema_reward) ** 2
            return reward / (np.sqrt(self._ema_var) + 1e-8)
        elif self.method == "baselines_return":
            # Order matters (verbatim from VecNormalize.step_wait): update the
            # accumulator with the RAW reward, fold it into the running stats,
            # normalize this step's reward, then zero the accumulator on done.
            st = self._brn_state
            st["ret"] = self.gamma * st["ret"] + reward
            delta = st["ret"] - st["mean"]
            tot = st["count"] + 1.0
            st["mean"] += delta / tot
            st["var"] = (st["var"] * st["count"] + delta * (st["ret"] - st["mean"])) / tot
            st["count"] = tot
            normed = reward / np.sqrt(st["var"] + 1e-8)
            if done:
                st["ret"] = 0.0
            return float(np.clip(normed, -self.clip, self.clip))
        else:
            return reward


class TerminateOnSuccessWrapper(gym.Wrapper):
    """Terminate the episode the moment ``info['success']`` is True."""

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("success", False):
            terminated = True
        return obs, reward, terminated, truncated, info


class GoalIterationWrapper(gym.Wrapper):
    """Cycle through MetaWorld goal positions sequentially on reset.

    Used for evaluation, where each task has 50 goals and we want to
    iterate through them deterministically.
    """

    def __init__(self, env: gym.Env, tasks: List) -> None:
        super().__init__(env)
        self._tasks = tasks
        self._task_idx = 0

    @property
    def num_goals(self) -> int:
        return len(self._tasks)

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.env.unwrapped.set_task(self._tasks[self._task_idx])
        self._task_idx = (self._task_idx + 1) % len(self._tasks)
        return self.env.reset(**kwargs)


def _make_single_env(
    env_name: str,
    task_cls: Any,
    task: Any,
    task_id: int,
    num_tasks: int,
    task_name: str,
    config: Config,
    goal_tasks: Optional[List] = None,
    randomize_goals: bool = False,
) -> gym.Env:
    """Build a single MetaWorld env with goal/normalization/one-hot wrappers."""
    env = task_cls(
        render_mode=None,
        reward_function_version=config.env.reward_function_version,
    )
    env.set_task(task)

    if goal_tasks is not None:
        if randomize_goals:
            # Training: randomly sample a goal on each reset.
            #
            # Reproducibility: both wrappers draw (or shuffle) goals with
            # env.np_random, which --seed does NOT seed.  set_seed() covers
            # random / numpy / torch and metaworld.MT*(seed=) fixes each
            # task's goal set, but the order in which training episodes visit
            # those goals comes from OS entropy in each env worker, so two
            # runs with the same --seed see different goal sequences.
            # (SawyerXYZEnv.reset(seed=) ignores its argument; only
            # env.unwrapped.seed(<per-env seed>) would seed this.)  The
            # published runs were produced with this behavior, so it is kept:
            # runs reproduce statistically (as seed-aggregate statistics:
            # mean ± std, or IQM ± std on V1, as the paper reports them), not
            # bit for bit.  Evaluation is unaffected (GoalIterationWrapper below).
            from metaworld.wrappers import RandomTaskSelectWrapper, PseudoRandomTaskSelectWrapper
            if config.env.task_select == "random":
                env = RandomTaskSelectWrapper(env, tasks=goal_tasks, sample_tasks_on_reset=True)
            elif config.env.task_select == "pseudorandom":
                env = PseudoRandomTaskSelectWrapper(env, tasks=goal_tasks, sample_tasks_on_reset=True)
            else:
                raise ValueError(f"Unknown task_select value: {config.env.task_select}")
        else:
            # Testing: deterministic goal iteration.
            env = GoalIterationWrapper(env, tasks=goal_tasks)

    if config.env.max_episode_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=config.env.max_episode_steps)

    if config.env.terminate_on_success:
        env = TerminateOnSuccessWrapper(env)

    # Training envs only: evaluation must report raw env rewards (success
    # metrics are scale-invariant, but test reward curves and best-checkpoint
    # selection should not depend on drifting normalizer statistics).
    if config.env.reward_normalization_method is not None and randomize_goals:
        env = RewardNormalizationWrapper(
            env,
            method=config.env.reward_normalization_method,
            gamma=config.training.gamma,
            clip=getattr(config.env, "reward_norm_clip", 10.0),
        )

    if config.env.use_one_hot:
        env = TaskIDWrapper(env, task_id=task_id, num_tasks=num_tasks, task_name=task_name)

    return env


def make_metaworld_envs(
    config: Config,
    seed: Optional[int] = None,
) -> Tuple[Any, Optional[Any], Dict[str, int], List[str], int, Dict[str, List[int]]]:
    """Build vectorized MetaWorld train/test envs (ShmemVectorEnv).

    Returns ``(train_envs, test_envs, task_name_to_id, task_names,
    num_goals_per_task, task_env_map)``.
    """
    import metaworld
    from tianshou.env.venvs import ShmemVectorEnv

    benchmark_seed = seed if seed is not None else config.seed
    if config.env.env_name == "MT10":
        benchmark = metaworld.MT10(seed=benchmark_seed)
    elif config.env.env_name == "MT25":
        benchmark = metaworld.MT25(seed=benchmark_seed)
    elif config.env.env_name == "MT50":
        benchmark = metaworld.MT50(seed=benchmark_seed)
    else:
        raise ValueError(f"Unknown benchmark: {config.env.env_name}. Use MT10, MT25, or MT50.")

    train_classes = benchmark.train_classes
    train_tasks = benchmark.train_tasks

    task_names = sorted(train_classes.keys())
    task_name_to_id = {name: i for i, name in enumerate(task_names)}
    num_tasks = len(task_names)

    if config.env.num_tasks != num_tasks:
        print(
            f"[MetaWorld] Config num_tasks={config.env.num_tasks} overridden to "
            f"benchmark num_tasks={num_tasks}"
        )

    per_task_train_tasks: Dict[str, List] = {name: [] for name in task_names}
    for task in train_tasks:
        per_task_train_tasks[task.env_name].append(task)

    envs_per_task = getattr(config.env, "envs_per_task", 1)
    if envs_per_task > 1:
        desired = num_tasks * envs_per_task
        if config.env.num_envs != desired:
            print(
                f"[MetaWorld] Config num_envs={config.env.num_envs} overridden to "
                f"num_tasks × envs_per_task = {num_tasks} × {envs_per_task} = {desired}"
            )
            config.env.num_envs = desired
    elif config.env.num_envs % num_tasks != 0:
        print(
            f"[MetaWorld] Warning: num_envs={config.env.num_envs} is not a multiple "
            f"of num_tasks={num_tasks}; tasks will receive uneven env counts via round-robin."
        )

    train_env_fns = []
    num_envs = config.env.num_envs
    task_env_map: Dict[str, List[int]] = {name: [] for name in task_names}

    # Round-robin: when num_envs = k · num_tasks, each task gets k envs.
    for i in range(num_envs):
        task_name = task_names[i % num_tasks]
        task_cls = train_classes[task_name]
        task_id = task_name_to_id[task_name]
        tasks_for_this = per_task_train_tasks[task_name]
        task = tasks_for_this[0]

        all_goals = per_task_train_tasks[task_name]

        def _make_env(
            _cls=task_cls,
            _task=task,
            _id=task_id,
            _name=task_name,
            _nt=num_tasks,
            _goals=all_goals,
        ):
            return _make_single_env(
                config.env.env_name, _cls, _task, _id, _nt, _name, config,
                goal_tasks=_goals, randomize_goals=True,
            )

        train_env_fns.append(_make_env)
        task_env_map[task_name].append(i)

    train_envs = ShmemVectorEnv(train_env_fns)

    num_goals_per_task = max(
        len(per_task_train_tasks[name]) for name in task_names
    ) if task_names else 50

    # Test envs: one per task, deterministic goal iteration.
    test_envs = None
    if config.logging.test_interval > 0:
        test_env_fns = []
        for task_name in task_names:
            task_cls = train_classes[task_name]
            task_id = task_name_to_id[task_name]
            all_goals = per_task_train_tasks[task_name]
            initial_task = all_goals[0]

            def _make_test_env(
                _cls=task_cls,
                _task=initial_task,
                _id=task_id,
                _name=task_name,
                _nt=num_tasks,
                _goals=all_goals,
            ):
                return _make_single_env(
                    config.env.env_name, _cls, _task, _id, _nt, _name, config,
                    goal_tasks=_goals,
                )

            test_env_fns.append(_make_test_env)

        test_envs = ShmemVectorEnv(test_env_fns)

    return train_envs, test_envs, task_name_to_id, task_names, num_goals_per_task, task_env_map


def make_single_render_env(
    config: Config,
    task_name: str,
    task_name_to_id: Dict[str, int],
    num_tasks: int,
    camera_name: str | None = None,
) -> gym.Env:
    """Single MetaWorld env with ``render_mode="rgb_array"`` for video rendering."""
    import metaworld

    if config.env.env_name == "MT10":
        benchmark = metaworld.MT10(seed=config.seed)
    elif config.env.env_name == "MT25":
        benchmark = metaworld.MT25(seed=config.seed)
    elif config.env.env_name == "MT50":
        benchmark = metaworld.MT50(seed=config.seed)
    else:
        raise ValueError(f"Unknown benchmark: {config.env.env_name}")

    train_classes = benchmark.train_classes
    train_tasks = benchmark.train_tasks

    if task_name not in train_classes:
        raise ValueError(
            f"Task '{task_name}' not found. Available: {sorted(train_classes.keys())}"
        )

    task_cls = train_classes[task_name]
    task_id = task_name_to_id[task_name]

    env_task = next((t for t in train_tasks if t.env_name == task_name), None)
    if env_task is None:
        raise ValueError(f"No task found for '{task_name}'")

    if camera_name:
        env = task_cls(render_mode="rgb_array", camera_name=camera_name)
    else:
        env = task_cls(render_mode="rgb_array")
    env.set_task(env_task)

    if config.env.max_episode_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=config.env.max_episode_steps)

    if config.env.use_one_hot:
        env = TaskIDWrapper(env, task_id=task_id, num_tasks=num_tasks, task_name=task_name)

    return env
