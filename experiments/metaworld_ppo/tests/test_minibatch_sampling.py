import unittest

import numpy as np
from tianshou.data.batch import Batch

from mtrl.algos.multitask_ppo import MultiTaskPPO


def _build_algo(minibatch_sampling: str, num_tasks: int) -> MultiTaskPPO:
    algo = object.__new__(MultiTaskPPO)
    algo._minibatch_sampling = minibatch_sampling
    algo._num_tasks = num_tasks
    return algo


def _build_batch(num_tasks: int, samples_per_task: list[int]) -> Batch:
    obs = []
    act = []
    adv = []
    logp_old = []
    returns = []
    v_s = []
    marker = []

    for task_id, count in enumerate(samples_per_task):
        one_hot = np.zeros(num_tasks, dtype=np.float32)
        one_hot[task_id] = 1.0
        for local_idx in range(count):
            raw_obs = np.array([task_id, local_idx], dtype=np.float32)
            obs.append(np.concatenate([raw_obs, one_hot]))
            act.append(np.array([local_idx], dtype=np.float32))
            adv.append(float(local_idx))
            logp_old.append(float(local_idx))
            returns.append(float(local_idx))
            v_s.append(float(local_idx))
            marker.append(task_id * 1000 + local_idx)

    return Batch(
        obs=np.asarray(obs, dtype=np.float32),
        act=np.asarray(act, dtype=np.float32),
        adv=np.asarray(adv, dtype=np.float32),
        logp_old=np.asarray(logp_old, dtype=np.float32),
        returns=np.asarray(returns, dtype=np.float32),
        v_s=np.asarray(v_s, dtype=np.float32),
        marker=np.asarray(marker, dtype=np.int64),
    )


class StratifiedMinibatchSamplingTest(unittest.TestCase):
    def test_stratified_minibatches_cover_all_tasks_with_expected_quota(self) -> None:
        np.random.seed(0)
        algo = _build_algo("stratified", num_tasks=10)
        batch = _build_batch(10, [6] * 10)

        minibatches = list(algo._iter_minibatches(batch, batch_size=13))

        self.assertEqual(len(minibatches), 3)
        expected_counts = np.array([2, 2, 2] + [1] * 7)
        for minibatch in minibatches:
            task_ids = minibatch.obs[:, -10:].argmax(axis=-1)
            counts = np.bincount(task_ids, minlength=10)
            np.testing.assert_array_equal(counts, expected_counts)

    def test_stratified_sampling_is_without_replacement_per_repeat(self) -> None:
        np.random.seed(1)
        algo = _build_algo("stratified", num_tasks=10)
        batch = _build_batch(10, [5] * 10)

        minibatches = list(algo._iter_minibatches(batch, batch_size=10))

        seen_markers = []
        for minibatch in minibatches:
            seen_markers.extend(minibatch.marker.tolist())
        self.assertEqual(len(seen_markers), len(set(seen_markers)))

    def test_stratified_sampling_stops_when_any_task_runs_out(self) -> None:
        np.random.seed(2)
        algo = _build_algo("stratified", num_tasks=10)
        batch = _build_batch(10, [3] * 9 + [1])

        minibatches = list(algo._iter_minibatches(batch, batch_size=10))

        self.assertEqual(len(minibatches), 1)
        task_ids = minibatches[0].obs[:, -10:].argmax(axis=-1)
        np.testing.assert_array_equal(np.bincount(task_ids, minlength=10), np.ones(10, dtype=np.int64))

    def test_stratified_sampling_requires_batch_size_to_cover_all_tasks(self) -> None:
        algo = _build_algo("stratified", num_tasks=10)
        batch = _build_batch(10, [2] * 10)

        with self.assertRaises(ValueError):
            list(algo._iter_minibatches(batch, batch_size=9))

    def test_mixed_sampling_delegates_to_batch_split(self) -> None:
        np.random.seed(3)
        algo = _build_algo("mixed", num_tasks=10)
        batch = _build_batch(10, [4] * 10)

        minibatches = list(algo._iter_minibatches(batch, batch_size=10))

        self.assertEqual(sum(len(minibatch) for minibatch in minibatches), len(batch))
        self.assertEqual(len(minibatches), 4)


if __name__ == "__main__":
    unittest.main()
