import numpy as np
import pickle as pickle
import os
from multiprocessing import Process, Pipe
from gymnasium.spaces import Box
from typing import Tuple, Optional, Sequence, Dict
InfoDict = Dict[str, float]


class MultiIterativeEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Internally, the environments are executed iteratively.

    Args:
        env (meta_policy_search.envs.base.MetaEnv): meta environment object
        meta_batch_size (int): number of meta tasks
        envs_per_task (int): number of environments per meta task
        max_path_length (int): maximum length of sampled environment paths - if the max_path_length is reached,
                               the respective environment is reset
    """

    def __init__(self, envs, max_path_length):
        self.envs = envs
        self.ts = np.zeros(len(self.envs), dtype="int")  # time steps
        self.max_path_length = max_path_length

    def step(self, actions):
        """
        Steps the wrapped environments with the provided actions

        Args:
            actions (list): lists of actions, of length meta_batch_size x envs_per_task

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool),
             env_infos (dict). Each list is of length meta_batch_size x envs_per_task
             (assumes that every task has same number of envs)
        """
        assert len(actions) == self.num_envs

        all_results = [env.step(a) for (a, env) in zip(actions, self.envs)]

        # stack results split to obs, rewards, ...
        obs, rewards, dones, env_infos = list(map(list, zip(*all_results)))

        # reset env when done or max_path_length reached
        dones = np.asarray(dones)
        self.ts += 1
        dones = np.logical_or(self.ts >= self.max_path_length, dones)
        for i in np.argwhere(dones).flatten():
            obs[i] = self.envs[i].reset()
            self.ts[i] = 0

        return obs, rewards, dones, env_infos

    def set_tasks(self, tasks):
        """
        Sets a list of tasks to each environment

        Args:
            tasks (list): list of the tasks for each environment
        """
        envs_per_task = np.split(self.envs, len(tasks))
        for task, envs in zip(tasks, envs_per_task):
            for env in envs:
                env.set_task(task)

    def reset(self):
        """
        Resets the environments

        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        obses = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return obses

    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
        return len(self.envs)


class MultiParallelEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Thereby the environments are distributed among meta_batch_size processes and
    executed in parallel.

    Args:
        env (meta_policy_search.envs.base.MetaEnv): meta environment object
        meta_batch_size (int): number of meta tasks
        envs_per_task (int): number of environments per meta task
    """

    def __init__(
        self,
        envs,
        num_tasks_per_env=1,
        tasks=None,
        normalize_obs=False,
        normalize_reward=False,
        obs_alpha=0.001,
        reward_alpha=0.001,
        seed=1,
        use_task_id=True,
    ):
        self.envs = envs
        self.n_tasks = num_tasks_per_env
        self.n_envs = len(envs)

        if tasks is None:
            self.parallel_envs = len(envs)
        else:
            self.parallel_envs = self.n_envs
        self.tasks = tasks
        self.use_task_id = use_task_id

        if self.n_envs > 1 and self.use_task_id:
            self._observation_space = Box(
                np.append(envs[0].observation_space.low, np.zeros(self.n_envs)),
                np.append(envs[0].observation_space.high, np.ones(self.n_envs)),
                dtype=float,
            )
        else:
            self._observation_space = Box(
                envs[0].observation_space.low,
                envs[0].observation_space.high,
                dtype=float,
            )
        self._action_space = envs[0].action_space

        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward

        self._obs_alpha = obs_alpha
        self._reward_alpha = reward_alpha

        obs_dim = np.prod(envs[0].observation_space.shape)
        self._obs_mean = np.zeros((self.n_envs, obs_dim))
        self._obs_var = np.zeros((self.n_envs, obs_dim))

        self._reward_mean = np.zeros(self.n_envs)
        self._reward_var = np.zeros(self.n_envs)

        self.remotes, self.work_remotes = zip(
            *[Pipe() for _ in range(self.parallel_envs)]
        )
        seeds = [seed + i + 1 for i in range(self.parallel_envs)]

        if self.tasks is None:
            self.ps = [
                Process(
                    target=worker, args=(work_remote, remote, pickle.dumps(env), seed)
                )
                for (env, work_remote, remote, seed) in zip(
                    self.envs, self.work_remotes, self.remotes, seeds
                )
            ]  # Why pass work remotes?
        else:
            self.ps = [
                Process(
                    target=worker,
                    args=(
                        work_remote,
                        remote,
                        pickle.dumps(env),
                        seed,
                        self.n_tasks,
                        tasks,
                    ),
                )
                for (env, work_remote, remote, seed, tasks) in zip(
                    self.envs,
                    self.work_remotes,
                    self.remotes,
                    seeds,
                    self.tasks.values(),
                )
            ]  # Why pass work remotes?

        for p in self.ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            p.start()
        for remote in self.work_remotes:
            remote.close()

        # MetaWorld 3 lazily builds MuJoCo models on first use. Building 50
        # MjModels concurrently can crash inside libc++; warm them in bounded
        # batches before JAX is imported. This does not consume reset RNG.
        if self.envs and hasattr(self.envs[0], "_ensure_env"):
            build_parallelism = max(1, int(os.environ.get("MUJOCO_BUILD_PARALLELISM", "4")))
            for start in range(0, len(self.remotes), build_parallelism):
                batch = self.remotes[start:start + build_parallelism]
                for remote in batch:
                    remote.send(("build", None))
                for offset, remote in enumerate(batch):
                    idx = start + offset
                    if not remote.poll(float(os.environ.get("MUJOCO_BUILD_TIMEOUT", "60"))):
                        proc = self.ps[idx]
                        raise RuntimeError(
                            f"MuJoCo worker {idx} did not build in time "
                            f"(alive={proc.is_alive()}, exitcode={proc.exitcode})"
                        )
                    remote.recv()
                    if os.environ.get("MUJOCO_BUILD_VERBOSE") == "1":
                        print(f"[mujoco-init] worker {idx + 1}/ {len(self.remotes)} ready", flush=True)

        if self.n_tasks > 1 and self.tasks is not None:
            self.set_tasks(self.tasks.values())

    def step(
        self, actions: list[np.ndarray]
    ) -> Tuple[list[np.ndarray], list[float], list[bool], list[InfoDict]]:
        assert len(actions) == self.num_envs * self.n_tasks or self.num_envs == 1

        # split list of actions in list of list of actions per meta tasks
        def chunks(l, n):
            return [l[x : x + n] for x in range(0, len(l), n)]

        actions_per_meta_task = chunks(actions, len(actions) // self.parallel_envs)

        # step remote environments
        for remote, action_list in zip(self.remotes, actions_per_meta_task):
            remote.send(("step", action_list))

        results = [remote.recv() for remote in self.remotes]

        obs, rewards, dones, env_infos = map(lambda x: sum(x, []), zip(*results))

        if self._normalize_obs:
            obs = self._apply_normalize_obs(obs)
        if self._normalize_reward:
            rewards = self._apply_normalize_reward(rewards)
        if self.n_envs > 1 and self.use_task_id:
            obs = self._add_task_id(obs)
        return obs, rewards, dones, env_infos

    def reset(self) -> list[np.ndarray]:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = sum([remote.recv() for remote in self.remotes], [])
        if self._normalize_obs:
            obs = self._apply_normalize_obs(obs)
        if self.n_envs > 1 and self.use_task_id:
            obs = self._add_task_id(obs)
        return obs

    def set_tasks(self, tasks: Optional[list[dict]] = None):
        # def chunks(l, n): return [l[x: x + n] for x in range(0, len(l), n)]
        # tasks_per_meta_task = chunks(tasks, self.n_tasks)
        tasks_per_meta_task = tasks
        import pdb

        for remote, task in zip(self.remotes, tasks_per_meta_task):
            remote.send(("set_task", task))

        results = [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))

    @property
    def observation_space(self) -> Box:
        return self._observation_space

    @property
    def action_space(self) -> Box:
        return self._action_space

    @property
    def num_envs(self) -> int:
        return self.n_envs

    def _update_obs_estimate(self, obs: list[np.ndarray]):
        for i, ob in enumerate(obs):
            self._obs_mean[i // self.n_tasks] = (1 - self._obs_alpha) * self._obs_mean[
                i // self.n_tasks
            ] + self._obs_alpha * ob
            self._obs_var[i // self.n_tasks] = (1 - self._obs_alpha) * self._obs_var[
                i // self.n_tasks
            ] + self._obs_alpha * np.square(ob - self._obs_mean[i // self.n_tasks])

    def _update_reward_estimate(self, rewards: list[np.ndarray]):
        for i, reward in enumerate(rewards):
            self._reward_mean[i // self.n_tasks] = (
                1 - self._reward_alpha
            ) * self._reward_mean[i // self.n_tasks] + self._reward_alpha * reward
            self._reward_var[i // self.n_tasks] = (
                1 - self._reward_alpha
            ) * self._reward_var[i // self.n_tasks] + self._reward_alpha * np.square(
                reward - self._reward_mean[i // self.n_tasks]
            )

    def _apply_normalize_obs(self, obs: list[np.ndarray]) -> list[np.ndarray]:
        self._update_obs_estimate(obs)
        return [
            (ob - self._obs_mean[i // self.n_tasks])
            / (np.sqrt(self._obs_var[i // self.n_tasks]) + 1e-8)
            for i, ob in enumerate(obs)
        ]

    def _apply_normalize_reward(self, rewards: list[np.ndarray]) -> list[np.ndarray]:
        self._update_reward_estimate(rewards)
        return [
            reward / (np.sqrt(self._reward_var[i // self.n_tasks]) + 1e-8)
            for i, reward in enumerate(rewards)
        ]

    def _add_task_id(self, observations: list[np.ndarray]) -> list[np.ndarray]:
        return [
            np.append(observation, np.eye(self.n_envs)[i // self.n_tasks])
            for i, observation in enumerate(observations)
        ]


class MultiParallelEvalEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Thereby the environments are distributed among meta_batch_size processes and
    executed in parallel.

    Args:
        env (meta_policy_search.envs.base.MetaEnv): meta environment object
        meta_batch_size (int): number of meta tasks
        envs_per_task (int): number of environments per meta task
    """

    def __init__(
        self,
        envs,
        normalize_obs=False,
        normalize_reward=False,
        obs_alpha=0.001,
        reward_alpha=0.001,
        seed=1,
    ):
        self.envs = envs
        self.n_envs = len(envs)
        self.active_tasks = np.arange(self.n_envs)

        self._observation_space = Box(
            envs[0].observation_space.low, envs[0].observation_space.high, dtype=float
        )
        self._action_space = envs[0].action_space

        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward

        self._obs_alpha = obs_alpha
        self._reward_alpha = reward_alpha

        obs_dim = np.prod(envs[0].observation_space.shape)
        self._obs_mean = np.zeros(obs_dim)
        self._obs_var = np.zeros(obs_dim)

        self._reward_mean = 0.0
        self._reward_var = 0.0

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])
        seeds = [seed + i + 1 for i in range(self.n_envs)]

        self.ps = [
            Process(target=worker, args=(work_remote, remote, pickle.dumps(env), seed))
            for (env, work_remote, remote, seed) in zip(
                self.envs, self.work_remotes, self.remotes, seeds
            )
        ]  # Why pass work remotes?

        for p in self.ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(
        self, actions: list[np.ndarray]
    ) -> Tuple[list[np.ndarray], list[float], list[bool], list[InfoDict]]:
        assert len(actions) == len(self.active_tasks) or len(self.active_tasks) == 1

        active_indices = list(self.active_tasks)

        def chunks(l, n):
            return [l[x : x + n] for x in range(0, len(l), n)]

        actions_per_meta_task = chunks(actions, len(actions) // len(self.active_tasks))

        for i, active_env in enumerate(active_indices):
            self.remotes[active_env].send(("step", actions_per_meta_task[i]))

        # for i, remote in enumerate(self.remotes):
        #     if i in active_indices:
        #         remote.send(('step', actions[i]))
        #     else:
        #         continue

        results = []
        new_active_tasks = []

        for i, remote in enumerate(self.remotes):
            if i in active_indices:
                result = remote.recv()
                results.append(result)
                obs, _, dones, _ = result
                if not all(dones):
                    new_active_tasks.append(i)

        self.active_tasks = np.array(new_active_tasks)

        obs, rewards, dones, env_infos = map(lambda x: sum(x, []), zip(*results))
        # env_infos = {key: [d[key] for d in env_infos] for key in env_infos[0]}

        if self._normalize_obs:
            obs = self._apply_normalize_obs(obs)
        if self._normalize_reward:
            rewards = self._apply_normalize_reward(rewards)

        return obs, rewards, dones, env_infos, active_indices

    def reset(self) -> list[np.ndarray]:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = sum([remote.recv() for remote in self.remotes], [])
        if self._normalize_obs:
            obs = self._apply_normalize_obs(obs)
        self.active_tasks = np.arange(self.n_envs)
        return obs

    def set_tasks(self, tasks: Optional[list[dict]] = None):
        tasks_per_meta_task = tasks

        for remote, task in zip(self.remotes, tasks_per_meta_task):
            remote.send(("set_task", task))

        results = [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))

    @property
    def observation_space(self) -> Box:
        return self._observation_space

    @property
    def action_space(self) -> Box:
        return self._action_space

    @property
    def num_envs(self) -> int:
        return self.n_envs

    def _update_obs_estimate(self, obs: list[np.ndarray]):
        for ob in obs:
            self._obs_mean = (
                1 - self._obs_alpha
            ) * self._obs_mean + self._obs_alpha * ob
            self._obs_var = (
                1 - self._obs_alpha
            ) * self._obs_var + self._obs_alpha * np.square(ob - self._obs_mean)

    def _update_reward_estimate(self, rewards: list[np.ndarray]):
        for reward in rewards:
            self._reward_mean = (
                1 - self._reward_alpha
            ) * self._reward_mean + self._reward_alpha * reward
            self._reward_var = (
                1 - self._reward_alpha
            ) * self._reward_var + self._reward_alpha * np.square(
                reward - self._reward_mean
            )

    def _apply_normalize_obs(self, obs: list[np.ndarray]) -> list[np.ndarray]:
        self._update_obs_estimate(obs)
        return [(ob - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8) for ob in (obs)]

    def _apply_normalize_reward(self, rewards: list[np.ndarray]) -> list[np.ndarray]:
        self._update_reward_estimate(rewards)
        return [reward / (np.sqrt(self._reward_var) + 1e-8) for reward in rewards]


def worker(
    remote,
    parent_remote,
    env_pickle: pickle,
    seed: int,
    n_envs: int = 1,
    tasks: Optional[list[dict]] = None,
):
    """
    Instantiation of a parallel worker for collecting samples. It loops continually checking the task that the remote
    sends to it.
    """
    parent_remote.close()

    envs = [pickle.loads(env_pickle) for _ in range(n_envs)]
    np.random.seed(seed)
    tasks = tasks

    while True:
        # receive command and data from the remote
        cmd, data = remote.recv()

        # do a step in each of the environment of the worker
        if cmd == "step":
            all_results = [env.step(a) for (a, env) in zip(data, envs)]
            obs, rewards, dones, infos = map(list, zip(*all_results))
            remote.send((obs, rewards, dones, infos))

        # Build lazy native simulators without consuming reset RNG. The parent
        # sends this in bounded batches to avoid massive concurrent MuJoCo XML compilation.
        elif cmd == "build":
            for env in envs:
                ensure_env = getattr(env, "_ensure_env", None)
                if ensure_env is not None:
                    ensure_env()
            remote.send(0)

        # reset all the environments of the worker
        elif cmd == "reset":
            obs = [env.reset() for env in envs]
            remote.send(obs)

        # set the specified task for each of the environments of the worker
        elif cmd == "set_task":
            for task, env in zip(data, envs):
                env.set_task(task)
            remote.send(0)

        # close the remote and stop the worker
        elif cmd == "close":
            remote.close()
            break

        else:
            raise NotImplementedError
