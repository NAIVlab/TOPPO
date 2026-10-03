import os
import inspect
import random

import uuid
import numpy as np
import tqdm
import math
import pyrallis
import wandb
from dataclasses import dataclass, asdict

from learner_moore import Learner
from dataset_utils import MultiTaskReplayBuffer
import utils

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@dataclass
class Config:
    # wandb params
    project: str = "MTRL"
    group: str = ""
    name: str = ""
    # model params
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_dim: int = 400
    n_layer: int = 3
    discount: float = 0.99
    tau: float = 5e-3
    n_critic: int = 2
    n_expert: int = 4
    dropout_rate: float = -1.0  # -1.0 means no dropout
    activation: str = "tanh"
    # training params
    use_ars: bool = False
    batch_size: int = 100
    max_steps: int = int(2e6)
    log_interval: int = 1000
    n_reset: int = 4
    replay_buffer_size: int = int(1e6)
    initial_step: int = 1500
    warmup_step: int = 3000
    n_train_per_step: int = 1
    # env params
    domain: str = "metaworld"
    env_name: str = "MT10"
    max_path_length: int = 500
    # evaluation params
    eval_episodes: int = 1
    eval_interval: int = int(1e4)
    # general params
    seed: int = 10
    eval_seed: int = 42

    def __post_init__(self):
        self.group = f"{self.domain}-{self.env_name}(H={self.max_path_length})"
        reset_interval = math.ceil(self.max_steps / (1000 * (self.n_reset + 1))) * 1000
        if reset_interval % 1e5 == 0:
            r_interval = int(reset_interval / 1e5)
        else:
            r_interval = reset_interval / 1e5

        alg_name = "ARS(MOORE)" if self.use_ars else "MOORE"

        self.name = (
            f"{alg_name}-NReset{self.n_reset}-Interval{r_interval}e5"
            if self.use_ars
            else f"{alg_name}"
        )


def setup_envs(config: Config):
    envs, env_names, tasks = utils.make_benchmark(
        config.domain, config.env_name, config.seed, config.max_path_length, False
    )
    env_names = list(env_names)
    env = utils.wrappers.MultiParallelEnvExecutor(envs)
    action_dim = env.action_space.shape[0]
    n_envs = env.n_envs

    eval_tasks = {
        env_name: tasks_per_env[:10] for env_name, tasks_per_env in tasks.items()
    }
    eval_envs = utils.make_eval_benchmark(
        config.domain,
        config.env_name,
        config.eval_seed,
        eval_tasks,
        config.max_path_length,
    )
    eval_env = utils.wrappers.MultiParallelEnvExecutor(eval_envs, 10, eval_tasks)

    return env, eval_env, env_names, action_dim, n_envs, tasks


def main(config: Config):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )

    env, eval_env, env_names, action_dim, n_envs, tasks = setup_envs(config)

    kwargs_ = asdict(config)
    kwargs_["hidden_dims"] = tuple([config.hidden_dim for _ in range(config.n_layer)])
    learner_args = inspect.signature(Learner).parameters.keys()

    kwargs = {k: v for k, v in kwargs_.items() if k in learner_args}
    agent = Learner(
        observations=env.observation_space.sample()[np.newaxis][np.newaxis],
        actions=env.action_space.sample()[np.newaxis][np.newaxis],
        n_task=n_envs,
        **kwargs,
    )

    replay_buffer = MultiTaskReplayBuffer(
        env.observation_space,
        action_dim,
        config.replay_buffer_size or config.max_steps,
        n_envs,
    )

    observations, dones = env.reset(), False
    rew_scale = np.ones((n_envs, 1)) if config.use_ars else None
    success = np.zeros(n_envs)
    reset_interval = (
        math.ceil(config.max_steps / (1000 * (config.n_reset + 1))) * 1000
        if config.use_ars
        else config.max_steps + 1
    )

    for i in tqdm.tqdm(range(1, config.max_steps + 1), smoothing=0.1, disable=False):
        if i > config.initial_step:
            if agent is None:
                if config.use_ars:
                    mean_rews = np.mean(
                        replay_buffer.rewards[:, : replay_buffer.size], axis=-1
                    )
                    rew_scale = np.reshape(
                        np.max(mean_rews) / np.maximum(mean_rews, 0.01), (n_envs, 1)
                    )
                    utils.log_rew_scale(i, rew_scale, env_names)
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

        replay_buffer.insert(
            observations, actions, rewards, masks, dones_float, next_observations
        )

        observations = next_observations

        if all(dones):
            env.set_tasks(
                [random.sample(tasks_per_env, k=1) for tasks_per_env in tasks.values()]
            )
            observations, dones = env.reset(), False
            if i % config.log_interval:
                utils.log_episode_info(i, infos, env_names, config.env_name, success)
            success = np.zeros(n_envs)

        if i > config.batch_size and i > config.initial_step:
            sample_kwargs = {"rew_scale": rew_scale} if config.use_ars else {}
            for _ in range(config.n_train_per_step):
                batch = replay_buffer.sample(config.batch_size, **sample_kwargs)
                update_info = agent.update(
                    batch,
                    critic_only=(i <= config.warmup_step),
                )

            if i % config.log_interval == 0:
                utils.log_update_metrics(i, update_info, env_names)

        if i % config.eval_interval == 0 and i > config.initial_step:
            utils.run_eval(
                i,
                agent,
                eval_env,
                config.eval_episodes,
                n_envs,
                env_names,
                config.env_name,
            )

        if config.use_ars and i % reset_interval == 0 and i > config.batch_size:
            mean_rews = np.mean(replay_buffer.rewards[:, : replay_buffer.size], axis=-1)
            rew_scale = np.reshape(
                np.max(mean_rews) / np.maximum(mean_rews, 0.01), (n_envs, 1)
            )
            utils.log_rew_scale(i, rew_scale, env_names)
            agent = Learner(
                observations=env.observation_space.sample()[np.newaxis],
                actions=env.action_space.sample()[np.newaxis],
                n_task=n_envs,
                **kwargs,
            )
            for _ in range(int(1e3)):
                batch = replay_buffer.sample(config.batch_size, rew_scale=rew_scale)
                update_info = agent.update(batch)
    env.close()
    eval_env.close()
    wandb.finish()


if __name__ == "__main__":
    config = pyrallis.parse(Config)
    main()
