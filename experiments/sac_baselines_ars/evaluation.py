from typing import Dict

import flax.linen as nn
import gymnasium as gym  # type hint only
import numpy as np


def evaluates(
    eval_f,
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    results = eval_f(**kwargs)
    return results


def evaluate_n_envs(
    agent: nn.Module,
    env: gym.Env,
    num_episodes: int,
    n_envs: int = 1,
    distribution="det",
    tasks=None,
    params=None,
    random=False,
    f_ac=lambda x: x,
) -> Dict[str, float]:
    stats = [{"return": [], "length": []} for _ in range(n_envs)]

    if tasks is not None:
        num_episodes = len(tasks)

    for task_id in range(num_episodes):
        if tasks is not None:
            env.set_tasks(tasks[task_id])
        observations = env.reset()
        n_tasks = len(observations) // n_envs

        dones = [False for _ in range(len(observations))]
        success = [0.0 for _ in range(len(observations))]

        while not all(dones):
            if random:
                actions = [env.action_space.sample() for _ in range(len(observations))]
            else:
                actions = agent.sample_actions(
                    np.array(observations).reshape(n_envs, n_tasks, -1),
                    temperature=float(distribution != "det"),
                    distribution=distribution,
                    params=params,
                )
            actions = np.reshape(actions, (n_tasks * n_envs, -1))
            observations, _, dones, infos = env.step(f_ac(actions))
            if "success" in infos[0]:
                if "success" not in stats[0]:
                    for stat in stats:
                        stat.update({"success": []})
                for env_id, info in enumerate(infos):
                    success[env_id] += info["success"]

        for k in stats[0].keys():
            if k == "success":
                for i, stat in enumerate(stats):
                    for j in range(n_tasks):
                        stat[k].append(int(success[n_tasks * i + j] > 0))
            else:
                for i, stat in enumerate(stats):
                    for j in range(n_tasks):
                        stat[k].append(infos[n_tasks * i + j]["episode"][k])

    for stat in stats:
        _stat = {}
        for k, v in stat.items():
            stat[k] = np.mean(v)
            _stat[f"{k}-max"] = np.max(v)
            _stat[f"{k}-min"] = np.min(v)
        stat.update(_stat)

    return stats
