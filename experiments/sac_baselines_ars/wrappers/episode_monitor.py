import time

import gym
import numpy as np

from wrappers.common import TimeStep


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        if hasattr(self.env, "max_path_length"):
            self.max_path_length = self.env.max_path_length
        else:
            self.max_path_length = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.reward_std = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.reward_std += reward**2
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if (
            self.episode_length >= self.max_path_length
            and self.max_path_length > 0
            and not done
        ):
            done = True
            info["TimeLimit.truncated"] = True

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["rew_mean"] = self.reward_sum / self.episode_length
            info["episode"]["rew_std"] = (
                self.reward_std / self.episode_length - info["episode"]["rew_mean"] ** 2
            ) ** 0.5
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                if isinstance(self.ref_max_score, float):
                    info["episode"]["return"] = (
                        self.get_normalized_score(info["episode"]["return"]) * 100.0
                    )

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()
