"""Multi-task Collector with cross-collect episode accumulation.

Maintains per-env (reward, win, steps) accumulators across ``collect()``
calls and patches the cumulative values onto the episode batch before the
hook in train.py sees them, fixing episode-stat truncation when episodes
span multiple collects.

Designed for n-step collection (OnPolicyTrainer), where local env index
equals global env index.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from tianshou.data.collector import Collector, CollectActionBatchProtocol, EpisodeBatchProtocol
from tianshou.data.types import RolloutBatchProtocol


class MultiTaskCollector(Collector):
    """Collector with per-env reward/win/step accumulation across collects."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_cumulative_reward = np.zeros(self.env_num, dtype=np.float64)
        self._env_max_win = np.zeros(self.env_num, dtype=np.float64)
        self._env_cumulative_steps = np.zeros(self.env_num, dtype=np.int64)
        self._pending_done_env_ids: deque[int] = deque()

    def run_on_step_hook(
        self,
        action_batch: CollectActionBatchProtocol,
        rollout_batch: RolloutBatchProtocol,
    ) -> None:
        rew = rollout_batch.rew
        info = rollout_batch.info
        done = rollout_batch.done
        n = len(rew)

        for i in range(n):
            info_i = info[i] if hasattr(info, '__getitem__') else {}
            # info_i may be a dict or a Tianshou Batch; both support .get().
            has_get = hasattr(info_i, 'get')
            if has_get and info_i.get("raw_reward", None) is not None:
                self._env_cumulative_reward[i] += float(info_i["raw_reward"])
            else:
                self._env_cumulative_reward[i] += float(rew[i])
            self._env_cumulative_steps[i] += 1
            if has_get:
                self._env_max_win[i] = max(
                    self._env_max_win[i],
                    float(info_i.get("win", 0.0)),
                )

        done_ids = np.where(done)[0]
        self._pending_done_env_ids.extend(int(x) for x in done_ids)

        super().run_on_step_hook(action_batch, rollout_batch)

    def run_on_episode_done(
        self,
        episode_batch: EpisodeBatchProtocol,
    ) -> dict[str, np.ndarray] | None:
        """Patch cumulative stats onto the batch, call hook, then strip them.

        The strip prevents Tianshou's ``Batch.__len__`` (used downstream in
        ``update_at_episode_done``) from tripping over scalar attributes.
        """
        patched = False
        if self._pending_done_env_ids:
            env_id = self._pending_done_env_ids.popleft()
            episode_batch._cumulative_reward = self._env_cumulative_reward[env_id]
            episode_batch._cumulative_win = self._env_max_win[env_id]
            episode_batch._cumulative_steps = int(self._env_cumulative_steps[env_id])
            self._env_cumulative_reward[env_id] = 0.0
            self._env_max_win[env_id] = 0.0
            self._env_cumulative_steps[env_id] = 0
            patched = True

        result = super().run_on_episode_done(episode_batch)

        if patched:
            del episode_batch._cumulative_reward
            del episode_batch._cumulative_win
            del episode_batch._cumulative_steps

        return result
