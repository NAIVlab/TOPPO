import collections
from typing import Optional

# NOTE: upstream imports d4rl here but never uses it; dropped so the online
# multi-task path needs no offline-RL/mujoco_py stack.
import gymnasium as gym  # type hints only (spaces.Box)
import numpy as np
from tqdm import tqdm

Batch = collections.namedtuple(
    "Batch",
    [
        "observations",
        "actions",
        "rewards",
        "masks",
        "next_observations",
        "next_actions",
        "returns",
        "heads",
    ],
)


def normalize_reward(rewards, rew_mean, rew_std):
    return (rewards - rew_mean) / (rew_std + 1e-6)


class Dataset(object):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
        size: int,
        use_heads: bool = False,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size
        self.use_heads = use_heads
        if use_heads:
            self.heads = np.zeros_like(self.rewards, dtype=np.int32)
        self.returns = np.zeros_like(self.rewards)

    def sample(self, batch_size: int, use_next_actions: bool = False) -> Batch:
        if use_next_actions:
            indx = np.random.randint(self.size - 1, size=batch_size)
        else:
            indx = np.random.randint(self.size, size=batch_size)

        return Batch(
            observations=self.observations[indx],
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            masks=self.masks[indx],
            next_observations=self.next_observations[indx],
            heads=self.heads[indx] if self.use_heads else None,
            next_actions=self.actions[indx + 1] if use_next_actions else None,
            returns=self.returns[indx],
        )


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_dim: int,
        capacity: int,
        use_heads: bool = False,
    ):

        observations = np.empty(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity,), dtype=np.float32)
        masks = np.empty((capacity,), dtype=np.float32)
        dones_float = np.empty((capacity,), dtype=np.float32)
        next_observations = np.empty(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )
        super().__init__(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=masks,
            dones_float=dones_float,
            next_observations=next_observations,
            use_heads=use_heads,
            size=0,
        )

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset, num_samples: Optional[int]):
        assert (
            self.insert_index == 0
        ), "Can insert a batch online in an empty replay buffer."

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert (
            self.capacity >= num_samples
        ), "Dataset cannot be larger than the replay buffer capacity."

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        normalize_rew: bool = False,
        rew_mean: float = 0.0,
        rew_std: float = 1.0,
        use_next_actions: bool = False,
        rew_scale: float = 1.0,
    ) -> Batch:
        if use_next_actions:
            indx = np.random.randint(self.size - 1, size=batch_size)
        else:
            indx = np.random.randint(self.size, size=batch_size)

        indx = np.random.randint(self.size, size=batch_size)
        if normalize_rew:
            rewards = normalize_reward(self.rewards[indx], rew_mean, rew_std)
        else:
            rewards = self.rewards[indx]

        return Batch(
            observations=self.observations[indx],
            actions=self.actions[indx],
            rewards=rewards * rew_scale,
            masks=self.masks[indx],
            next_observations=self.next_observations[indx],
            next_actions=self.actions[indx + 1] if use_next_actions else None,
            heads=self.heads[indx] if self.use_heads else None,
            returns=self.returns[indx],
        )


class MultiTaskReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_dim: int,
        capacity: int,
        n_task: int = 1,
        use_heads: bool = False,
    ):

        # Store float64 obs as float32: JAX truncates f64 -> f32 at the jit
        # boundary anyway (x64 off), so the learner sees bit-identical values
        # while buffer RAM and checkpoint size halve (MT50@500k: 36 -> 18 GB).
        obs_dtype = (
            np.float32
            if observation_space.dtype == np.float64
            else observation_space.dtype
        )
        observations = np.empty(
            (n_task, capacity, *observation_space.shape), dtype=obs_dtype
        )
        actions = np.empty((n_task, capacity, action_dim), dtype=np.float32)
        rewards = np.empty((n_task, capacity), dtype=np.float32)
        masks = np.empty((n_task, capacity), dtype=np.float32)
        dones_float = np.empty((n_task, capacity), dtype=np.float32)
        next_observations = np.empty(
            (n_task, capacity, *observation_space.shape), dtype=obs_dtype
        )

        observations.fill(0.0)
        actions.fill(0.0)
        rewards.fill(0.0)
        masks.fill(0.0)
        dones_float.fill(0.0)
        next_observations.fill(0.0)

        super().__init__(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=masks,
            dones_float=dones_float,
            next_observations=next_observations,
            size=0,
            use_heads=use_heads,
        )

        self.size = 0
        self.n_task = n_task

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset, num_samples: Optional[int]):
        assert (
            self.insert_index == 0
        ), "Can insert a batch online in an empty replay buffer."

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert (
            self.capacity >= num_samples
        ), "Dataset cannot be larger than the replay buffer capacity."

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ):
        assert len(observation) == self.n_task
        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.dones_float[:, self.insert_index] = done_float
        self.next_observations[:, self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        norm_rew: bool = False,
        rew_mean: np.ndarray = np.zeros(1),
        rew_std: np.ndarray = np.ones(1),
        use_next_actions: bool = False,
        rew_scale=None,
    ) -> Batch:
        if rew_scale is None:
            rew_scale = np.ones((self.n_task, 1))
        if use_next_actions:
            indx = np.random.randint(self.size - 1, size=(self.n_task, batch_size))
        else:
            indx = np.random.randint(self.size, size=(self.n_task, batch_size))

        if norm_rew:
            rewards = normalize_reward(
                self.rewards[np.arange(self.n_task)[:, None], indx], rew_mean, rew_std
            )
        else:
            rewards = self.rewards[np.arange(self.n_task)[:, None], indx]

        return Batch(
            observations=self.observations[np.arange(self.n_task)[:, None], indx],
            actions=self.actions[np.arange(self.n_task)[:, None], indx],
            rewards=rewards * rew_scale,
            masks=self.masks[np.arange(self.n_task)[:, None], indx],
            next_observations=self.next_observations[
                np.arange(self.n_task)[:, None], indx
            ],
            next_actions=(
                self.actions[np.arange(self.n_task)[:, None], indx + 1]
                if use_next_actions
                else None
            ),
            heads=(
                self.heads[np.arange(self.n_task)[:, None], indx]
                if self.use_heads
                else None
            ),
            returns=self.returns[np.arange(self.n_task)[:, None], indx],
        )
