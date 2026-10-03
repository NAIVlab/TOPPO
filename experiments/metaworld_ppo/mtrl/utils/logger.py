from __future__ import annotations

import csv
import os
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch
from tianshou.data.stats import SequenceSummaryStats
from torch.utils.tensorboard import SummaryWriter


from tianshou.utils.logger.logger_base import BaseLogger, VALID_LOG_VALS_TYPE, DataScope

if TYPE_CHECKING:
    from mtrl.utils.config import Config
    from tianshou.policy import BasePolicy


# Keys that should be logged to CSV (prefix match)
_CSV_PREFIXES = (
    "reward/",
    "win_rate/",
    "test_reward/",
    "test_win_rate/",
    "test_episode/",
    "test/",
)

# Keys that should be logged to TensorBoard and WandB.
_TB_WANDB_PREFIXES = (
    "approx_kl/",
    "clip_fraction/",
    "explained_variance/",
    "data/",
    "episode_length/",
    "cagrad/",
    "cagrad_actor/",
    "cagrad_critic/",
    "fairgrad/",
    "fairgrad_actor/",
    "fairgrad_critic/",
    "pcgrad/",
    "pcgrad_actor/",
    "pcgrad_critic/",
    "popart/",
    "retscale/",
    "hlgauss/",
    "reward/",
    "win_rate/",
    "rollout/",
    "train/",
    "test_reward/",
    "test_win_rate/",
    "test_episode/",
    "test/",
    "grad/",
    "ploss/",
    "worst/",
)


class MTRLLogger(BaseLogger):
    """Multi-backend logger (WandB + TensorBoard + CSV).

    Layout:
        logs/<exp_name>/
        ├── config.yaml
        ├── events.out.tfevents.*
        ├── metric.csv
        └── checkpoints/

    W&B run files are written under ``$TMPDIR/wandb`` to keep the
    experiment directory clean; end-of-epoch info logs are skipped.
    """

    def __init__(
        self,
        config: Config,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        use_csv: bool = True,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        info_interval: int = 1,
        save_interval: int = 1000,
    ) -> None:
        super().__init__(
            training_interval=train_interval,
            test_interval=test_interval,
            update_interval=update_interval,
            info_interval=info_interval,
            save_interval=save_interval,
        )
        self.last_log_test_step = -1
        self.last_log_update_step = -1
        self.last_log_info_step = -1
        self.last_save_step = -1
        self._latest_env_step = 0
        self.config = config
        self._last_known_task_win_rate: Dict[str, float] = {}
        self._last_known_task_reward: Dict[str, float] = {}
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.use_csv = use_csv

        # Deterministic layout so re-entry / resume can find prior state.
        # Same exp_name re-uses the directory (checkpoints, wandb_run_id,
        # CSV all append); use a unique exp_name to disambiguate runs.
        self.exp_dir = os.path.join(config.logging.log_dir, config.exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        # Resume iff a rolling latest.pth exists — must match train.py's
        # CheckpointManager.find_latest() predicate exactly.  A weaker
        # predicate would silently append to an existing W&B run / CSV
        # while training actually cold-starts.
        self._is_resume = os.path.exists(os.path.join(self.ckpt_dir, "latest.pth"))
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        from mtrl.utils.config import save_config
        save_config(config, os.path.join(self.exp_dir, "config.yaml"))

        self._tb_writer: SummaryWriter | None = None
        self._wandb_run = None
        self._wandb_id_path = os.path.join(self.exp_dir, ".wandb_id")
        self._csv_file = None
        self._csv_writer = None
        self._csv_path = os.path.join(self.exp_dir, "metric.csv")
        self._csv_columns: List[str] = ["step"]

        if self.use_tensorboard:
            self._tb_writer = SummaryWriter(log_dir=self.exp_dir)

        # On resume, re-use the saved W&B run_id so history appends.
        if self.use_wandb:
            try:
                import wandb
                tmp_dir = os.environ.get("TMPDIR", "/tmp")
                wandb_dir = os.path.join(tmp_dir, "wandb")
                os.makedirs(wandb_dir, exist_ok=True)

                saved_run_id: Optional[str] = None
                if os.path.exists(self._wandb_id_path):
                    try:
                        with open(self._wandb_id_path) as f:
                            saved_run_id = f.read().strip() or None
                    except OSError:
                        saved_run_id = None

                init_kwargs: Dict[str, Any] = dict(
                    project=config.project_name,
                    name=config.exp_name,
                    group=config.logging.wandb_group,
                    tags=config.logging.wandb_tags or None,
                    notes=config.logging.wandb_notes,
                    mode=config.logging.wandb_mode,
                    config=asdict(config),
                    dir=wandb_dir,
                    reinit=True,
                )
                if saved_run_id:
                    init_kwargs["id"] = saved_run_id
                    init_kwargs["resume"] = "allow"

                self._wandb_run = wandb.init(**init_kwargs)

                # Persist run_id for next re-entry.
                if self._wandb_run is not None:
                    try:
                        with open(self._wandb_id_path, "w") as f:
                            f.write(str(self._wandb_run.id))
                    except OSError as e:
                        print(f"[MTRLLogger] Could not persist wandb_run_id: {e}")
            except Exception as e:
                print(f"[MTRLLogger] WandB init failed: {e}. Continuing without WandB.")
                self.use_wandb = False

        # On resume, append to the existing CSV preserving its column schema.
        if self.use_csv:
            existing_header: Optional[List[str]] = None
            if self._is_resume and os.path.exists(self._csv_path):
                try:
                    with open(self._csv_path) as f:
                        first = f.readline().strip()
                    if first:
                        existing_header = first.split(",")
                except OSError:
                    existing_header = None

            if existing_header:
                self._csv_columns = existing_header
                self._csv_file = open(self._csv_path, "a", newline="")
                self._csv_writer = csv.DictWriter(
                    self._csv_file,
                    fieldnames=self._csv_columns,
                    extrasaction="ignore",
                )
            else:
                self._csv_file = open(self._csv_path, "w", newline="")
                self._csv_writer = csv.DictWriter(
                    self._csv_file,
                    fieldnames=self._csv_columns,
                    extrasaction="ignore",
                )
                self._csv_writer.writeheader()

        self.algorithm: Optional[BasePolicy] = None

    def set_algorithm(self, algorithm: BasePolicy) -> None:
        self.algorithm = algorithm

    @staticmethod
    def _is_csv_key(key: str) -> bool:
        return any(key.startswith(p) for p in _CSV_PREFIXES)

    @staticmethod
    def _is_tb_wandb_key(key: str) -> bool:
        return any(key.startswith(p) for p in _TB_WANDB_PREFIXES)

    @staticmethod
    def _extract_first(data: Dict[str, float], candidates: tuple[str, ...]) -> float | None:
        for key in candidates:
            if key in data:
                return float(data[key])
        return None

    @staticmethod
    def _extract_per_task_metric(flat: Dict[str, float], metric_name: str) -> Dict[str, float]:
        """Extract per-task metrics matching ``<metric>/<task>``,
        ``per_task_<metric>/<task>``, or ``per_task_stats/<task>/<metric>``."""
        result: Dict[str, float] = {}
        direct_prefix = f"{metric_name}/"
        field_prefix = f"per_task_{metric_name}/"

        for key, value in flat.items():
            if key.startswith(direct_prefix):
                task = key[len(direct_prefix):]
                if task:
                    result[f"{metric_name}/{task}"] = float(value)
            elif key.startswith(field_prefix):
                task = key[len(field_prefix):]
                if task:
                    result[f"{metric_name}/{task}"] = float(value)

        per_task_prefix = "per_task_stats/"
        per_task_suffix = f"/{metric_name}"
        for key, value in flat.items():
            if key.startswith(per_task_prefix) and key.endswith(per_task_suffix):
                task = key[len(per_task_prefix):-len(per_task_suffix)]
                if task:
                    result[f"{metric_name}/{task}"] = float(value)

        return result

    def _get_learning_rate(self) -> float:
        if self.algorithm is not None:
            for attr in ("_optim", "optim", "policy_optim", "critic_optim", "critic2_optim"):
                optim = getattr(self.algorithm, attr, None)
                if optim and hasattr(optim, "param_groups") and optim.param_groups:
                    lr = optim.param_groups[0].get("lr", None)
                    if lr is not None:
                        return float(lr)
        return float(self.config.training.learning_rate)

    def _estimate_policy_std(self) -> float | None:
        """Estimate policy std from actor parameters when available."""
        if self.algorithm is None:
            return None

        policy = getattr(self.algorithm, "policy", None)
        actor = getattr(policy, "actor", None)
        if actor is None:
            return None

        sigma_param = getattr(actor, "sigma_param", None)
        if sigma_param is not None:
            sigma = torch.clamp(sigma_param.detach(), min=-20, max=2).exp()
            return float(sigma.mean().item())

        log_std = getattr(actor, "log_std", None)
        if log_std is not None and torch.is_tensor(log_std):
            return float(log_std.detach().exp().mean().item())

        return None

    def _resolve_backend_step(self, step_type: str, step: int) -> int:
        """Use env_step for update/info writes so W&B steps stay monotonic."""
        if step_type.endswith("/update_step") or step_type.endswith("/epoch"):
            return max(self._latest_env_step, step)
        return step

    def write(self, step_type: str, step: int, data: dict[str, VALID_LOG_VALS_TYPE]) -> None:
        """Write scalars to TB/WandB, and a filtered subset to CSV."""
        if step_type.endswith("/env_step"):
            self._latest_env_step = max(self._latest_env_step, step)
        backend_step = self._resolve_backend_step(step_type, step)

        wandb_payload: dict[str, Any] = {}
        for key, value in data.items():
            if (
                isinstance(value, (int, float, np.number))
                and self._is_tb_wandb_key(key)
            ):
                scalar_value = float(value)
                if self._tb_writer:
                    self._tb_writer.add_scalar(key, scalar_value, global_step=backend_step)
                if self._wandb_run:
                    wandb_payload[key] = scalar_value

        if self._wandb_run and wandb_payload:
            self._wandb_run.log(wandb_payload, step=backend_step)

        if self._tb_writer:
            self._tb_writer.flush()

        if self.use_csv and self._csv_file and self._csv_writer:
            row_data: dict[str, Any] = {"step": backend_step}
            for k, v in data.items():
                if isinstance(v, (int, float, np.number)) and self._is_csv_key(k):
                    row_data[k] = float(v)

            if len(row_data) > 1:
                if any(k not in self._csv_columns for k in row_data):
                    self._update_csv_schema(row_data)

                if self._csv_writer is not None:
                    self._csv_writer.writerow(row_data)
                    self._csv_file.flush()

    def _write_histograms(self, step: int) -> None:
        """Per-layer parameter + gradient histograms (gated; high-volume)."""
        if self.algorithm is None:
            return
        if not getattr(self.config.logging, "log_weights_grads_histograms", False):
            return

        hist_step = max(self._latest_env_step, step)
        for name, param in self.algorithm.named_parameters():
            if param.numel() > 0:
                if self._tb_writer:
                    self._tb_writer.add_histogram(f"parameters/{name}", param.data, global_step=hist_step)
            if param.grad is not None and param.grad.numel() > 0:
                if self._tb_writer:
                    self._tb_writer.add_histogram(
                        f"gradients/{name}",
                        param.grad.data,
                        global_step=hist_step,
                    )

        if self._tb_writer:
            self._tb_writer.flush()

        if self._wandb_run:
            import wandb
            histograms: dict[str, Any] = {}
            for name, param in self.algorithm.named_parameters():
                if param.numel() > 0:
                    histograms[f"parameters/{name}"] = wandb.Histogram(param.data.cpu().numpy())
                if param.grad is not None and param.grad.numel() > 0:
                    histograms[f"gradients/{name}"] = wandb.Histogram(param.grad.data.cpu().numpy())
            if histograms:
                self._wandb_run.log(histograms, step=hist_step)

    def _write_data_histograms(self, step: int) -> None:
        """Rollout data (values / returns / advantages) histograms + scalar stats."""
        if self.algorithm is None:
            return

        latest_data = getattr(self.algorithm, "_latest_data", None)
        if latest_data is None:
            return

        hist_step = max(self._latest_env_step, step)
        wandb_payload: dict[str, Any] = {}
        scalar_payload: dict[str, float] = {}

        for name, tensor in latest_data.items():
            if tensor.numel() == 0:
                continue

            arr = tensor.float().numpy()
            tag = f"data/{name}"

            if self._tb_writer:
                self._tb_writer.add_histogram(tag, arr, global_step=hist_step)

            if self._wandb_run:
                import wandb
                wandb_payload[tag] = wandb.Histogram(arr)

            scalar_payload[f"{tag}_mean"] = float(arr.mean())
            scalar_payload[f"{tag}_min"] = float(arr.min())
            scalar_payload[f"{tag}_max"] = float(arr.max())
            scalar_payload[f"{tag}_std"] = float(arr.std())

        if self._tb_writer:
            for key, val in scalar_payload.items():
                self._tb_writer.add_scalar(key, val, global_step=hist_step)
            self._tb_writer.flush()

        if self._wandb_run:
            wandb_payload.update(scalar_payload)
            if wandb_payload:
                self._wandb_run.log(wandb_payload, step=hist_step)

        self.algorithm._latest_data = None

    def _update_csv_schema(self, new_row: Dict[str, Any]) -> None:
        """Rewrite the CSV with an expanded column set.

        The close+reopen dance can fail on network filesystems and
        under concurrent overwrite; a failed schema extension must not
        kill a multi-hour run.  Fall back to the existing schema with
        ``extrasaction="ignore"`` (new columns silently dropped); if the
        append reopen also fails, disable CSV and let TB / W&B carry on.
        """
        new_keys = [k for k in new_row if k not in self._csv_columns]
        if not new_keys:
            return

        old_columns = list(self._csv_columns)
        try:
            self._csv_file.close()
            with open(self._csv_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))

            expanded = old_columns + new_keys
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=expanded,
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()
            self._csv_writer.writerows(rows)
            self._csv_file.flush()
            self._csv_columns = expanded
        except OSError as e:
            print(
                f"[MTRLLogger] CSV schema rewrite failed ({e}); keeping old "
                f"columns and dropping new keys {new_keys} going forward."
            )
            try:
                self._csv_file = open(self._csv_path, "a", newline="")
                self._csv_writer = csv.DictWriter(
                    self._csv_file,
                    fieldnames=old_columns,
                    extrasaction="ignore",
                )
            except OSError as e2:
                print(f"[MTRLLogger] CSV append reopen failed ({e2}); disabling CSV.")
                self._csv_file = None
                self._csv_writer = None
                self.use_csv = False

    def prepare_dict_for_logging(self, log_data: dict) -> dict[str, VALID_LOG_VALS_TYPE]:
        """Flatten nested dicts and keep only scalar numeric values."""

        def flatten(d: dict, parent_key: str = "", sep: str = "/") -> dict:
            items: list[tuple[str, Any]] = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key, sep=sep).items())
                elif isinstance(v, SequenceSummaryStats):
                    items.append((new_key, v.mean))
                else:
                    items.append((new_key, v))
            return dict(items)

        flat = flatten(log_data)
        return {k: v for k, v in flat.items() if isinstance(v, (int, float, np.number))}

    def _merge_task_stats_into(
        self,
        task_stats: Dict[str, Dict[str, Any]],
        merged: dict[str, Any],
    ) -> None:
        """Merge per-task episode stats into the logging dict.

        Skip tasks with zero episodes for per-task curves / rollout averages
        (to avoid false zeros); for ``worst/*`` carry the last-known value
        so tail tasks aren't hidden when a window happens to miss them.
        """
        mean_reward_values: list[float] = []
        mean_length_values: list[float] = []
        mean_win_values: list[float] = []

        for task, metrics in task_stats.items():
            if metrics.get("num_episodes", 0) == 0:
                continue
            for m_name in ("reward", "win_rate", "episode_length", "approx_kl", "clip_fraction"):
                if m_name in metrics:
                    value = float(metrics[m_name])
                    merged[f"{m_name}/{task}"] = value
                    if m_name == "reward":
                        mean_reward_values.append(value)
                        self._last_known_task_reward[task] = value
                    elif m_name == "episode_length":
                        mean_length_values.append(value)
                    elif m_name == "win_rate":
                        mean_win_values.append(value)
                        self._last_known_task_win_rate[task] = value

        if "rollout/mean_reward" not in merged and mean_reward_values:
            merged["rollout/mean_reward"] = float(np.mean(mean_reward_values))
        if "rollout/mean_episode_length" not in merged and mean_length_values:
            merged["rollout/mean_episode_length"] = float(np.mean(mean_length_values))
        if mean_win_values:
            merged["rollout/mean_win_rate"] = float(np.mean(mean_win_values))

        # Carry-forward for worst-task metrics.
        all_win_rates = dict(self._last_known_task_win_rate)
        all_rewards = dict(self._last_known_task_reward)
        if all_win_rates:
            merged["worst/success"] = float(min(all_win_rates.values()))
        if all_rewards:
            merged["worst/reward"] = float(min(all_rewards.values()))

    def log_training_data(self, log_data: dict, step: int) -> None:
        """Log per-task train metrics + rollout summaries."""
        if step - self.last_log_training_step < self.training_interval:
            return

        data = self.prepare_dict_for_logging(log_data)
        merged: dict[str, Any] = {}

        # Under env-side reward normalization the collector's reward stat is
        # in normalized-clipped units; skip it so _merge_task_stats_into
        # fills rollout/mean_reward from the raw per-task episode rewards
        # (the hooks prefer info["raw_reward"]) and the curve stays
        # comparable across normalizer arms.
        env_reward_normalized = (
            getattr(self.config.env, "reward_normalization_method", None) is not None
        )
        rew = self._extract_first(data, ("rew", "returns_stat/mean"))
        if rew is not None and not env_reward_normalized:
            merged["rollout/mean_reward"] = rew

        ep_len = self._extract_first(data, ("len", "lens_stat/mean"))
        if ep_len is not None:
            merged["rollout/mean_episode_length"] = ep_len

        if self.algorithm and hasattr(self.algorithm, "get_and_reset_per_task_train_stats"):
            task_stats = self.algorithm.get_and_reset_per_task_train_stats()
            self._merge_task_stats_into(task_stats, merged)

        merged.update(self._extract_per_task_metric(data, "approx_kl"))
        merged.update(self._extract_per_task_metric(data, "clip_fraction"))

        self.write(f"{DataScope.TRAINING}/env_step", step, merged)
        self.last_log_training_step = step

    def log_test_data(self, log_data: dict, step: int) -> None:
        """Log per-task test metrics."""
        if step - self.last_log_test_step < self.test_interval:
            return

        data = self.prepare_dict_for_logging(log_data)
        merged: dict[str, Any] = {}

        test_rew = self._extract_first(data, ("rew", "returns_stat/mean"))
        if test_rew is not None:
            merged["test/mean_reward"] = test_rew

        if self.algorithm and hasattr(self.algorithm, "get_and_reset_per_task_test_stats"):
            task_stats = self.algorithm.get_and_reset_per_task_test_stats()
            rewards: list[float] = []
            win_rates: list[float] = []
            ep_lengths: list[float] = []
            for task, metrics in task_stats.items():
                if "reward" in metrics:
                    rew_val = float(metrics["reward"])
                    merged[f"test_reward/{task}"] = rew_val
                    rewards.append(rew_val)
                if "win_rate" in metrics:
                    win = float(metrics["win_rate"])
                    merged[f"test_win_rate/{task}"] = win
                    win_rates.append(win)
                if "episode_length" in metrics:
                    ep_len = float(metrics["episode_length"])
                    merged[f"test_episode/{task}_length"] = ep_len
                    ep_lengths.append(ep_len)
                if "num_episodes" in metrics:
                    merged[f"test_episode/{task}_num_episodes"] = float(metrics["num_episodes"])

            if "test/mean_reward" not in merged and rewards:
                merged["test/mean_reward"] = float(np.mean(rewards))
            if win_rates:
                merged["test/mean_win_rate"] = float(np.mean(win_rates))
            if ep_lengths:
                merged["test/mean_episode_length"] = float(np.mean(ep_lengths))

        merged.update(self._extract_per_task_metric(data, "test_reward"))
        merged.update(self._extract_per_task_metric(data, "test_win_rate"))

        self.write(f"{DataScope.TEST}/env_step", step, merged)
        self.last_log_test_step = step

    def log_update_data(self, log_data: dict, step: int) -> None:
        """Log training losses, then gradients/parameters as histograms."""
        if step - self.last_log_update_step < self.update_interval:
            return

        data = self.prepare_dict_for_logging(log_data)
        merged: dict[str, float] = {}

        loss = self._extract_first(data, ("loss", "loss/mean", "smoothed_loss/loss"))
        policy_loss = self._extract_first(
            data,
            (
                "policy_loss",
                "policy_loss/mean",
                "actor_loss",
                "actor_loss/mean",
                "smoothed_loss/policy_loss",
                "smoothed_loss/actor_loss",
                "loss/policy",
                "loss/actor",
            ),
        )
        value_loss = self._extract_first(
            data,
            (
                "value_loss",
                "value_loss/mean",
                "vf_loss",
                "vf_loss/mean",
                "smoothed_loss/value_loss",
                "smoothed_loss/vf_loss",
                "loss/vf",
                "loss/value",
            ),
        )
        entropy = self._extract_first(
            data,
            (
                "entropy", 
                "entropy/mean", 
                "ent_loss", 
                "ent_loss/mean", 
                "smoothed_loss/entropy", 
                "smoothed_loss/ent_loss",
                "loss/ent",
                "loss/entropy",
            ),
        )
        approx_kl = self._extract_first(
            data,
            ("approx_kl", "approx_kl/mean", "smoothed_loss/approx_kl", "loss/kl"),
        )
        clip_fraction = self._extract_first(
            data,
            ("clip_fraction", "clip_fraction/mean", "smoothed_loss/clip_fraction", "clip_loss", "loss/clip", "loss/clip_loss"),
        )
        explained_variance = self._extract_first(
            data,
            ("explained_variance", "explained_variance/mean", "smoothed_loss/explained_variance"),
        )
        std = self._extract_first(
            data,
            ("std", "std/mean", "smoothed_loss/std"),
        )
        kl = self._extract_first(
            data,
            ("kl", "kl/mean", "smoothed_loss/kl"),
        )
        step_size = self._extract_first(
            data,
            ("step_size", "step_size/mean", "smoothed_loss/step_size"),
        )

        if std is None:
            std = self._estimate_policy_std()

        if loss is not None:
            merged["train/loss"] = loss
        if policy_loss is not None:
            merged["train/policy_loss"] = policy_loss
        if value_loss is not None:
            merged["train/value_loss"] = value_loss
        if entropy is not None:
            merged["train/entropy"] = entropy
        if approx_kl is not None:
            merged["train/approx_kl"] = approx_kl
        if clip_fraction is not None:
            merged["train/clip_fraction"] = clip_fraction
        if explained_variance is not None:
            merged["train/explained_variance"] = explained_variance
        if std is not None:
            merged["train/std"] = std
        if kl is not None:
            merged["train/kl"] = kl
        if step_size is not None:
            merged["train/step_size"] = step_size

        if self.config.algo == "ppo":
            merged["train/clip_range"] = float(self.config.ppo.clip_range)
        merged["train/learning_rate"] = self._get_learning_rate()

        if self.algorithm and hasattr(self.algorithm, "get_and_reset_per_task_update_stats"):
            task_stats = self.algorithm.get_and_reset_per_task_update_stats()
            for task, metrics in task_stats.items():
                for m_name, val in metrics.items():
                    merged[f"{m_name}/{task}"] = val

        # Must run AFTER get_and_reset_per_task_update_stats — that flush
        # populates the cache this drains.
        if self.algorithm and hasattr(self.algorithm, "get_and_reset_grad_aggregates"):
            merged.update(self.algorithm.get_and_reset_grad_aggregates())

        if self.algorithm and hasattr(self.algorithm, "get_and_reset_pcgrad_update_stats"):
            merged.update(self.algorithm.get_and_reset_pcgrad_update_stats())

        if self.algorithm and hasattr(self.algorithm, "get_and_reset_cagrad_update_stats"):
            merged.update(self.algorithm.get_and_reset_cagrad_update_stats())

        if self.algorithm and hasattr(self.algorithm, "get_and_reset_fairgrad_update_stats"):
            merged.update(self.algorithm.get_and_reset_fairgrad_update_stats())

        if self.algorithm and hasattr(self.algorithm, "get_and_reset_popart_diagnostics"):
            merged.update(self.algorithm.get_and_reset_popart_diagnostics())

        if self.algorithm and hasattr(self.algorithm, "get_and_reset_return_scaling_diagnostics"):
            merged.update(self.algorithm.get_and_reset_return_scaling_diagnostics())

        if self.algorithm and hasattr(self.algorithm, "get_and_reset_hlgauss_diagnostics"):
            merged.update(self.algorithm.get_and_reset_hlgauss_diagnostics())

        merged.update(self._extract_per_task_metric(data, "approx_kl"))
        merged.update(self._extract_per_task_metric(data, "clip_fraction"))
        merged.update(self._extract_per_task_metric(data, "explained_variance"))

        self.write(f"{DataScope.UPDATE}/update_step", step, merged)
        self._write_histograms(step)
        self._write_data_histograms(step)

        self.last_log_update_step = step

    def log_info_data(self, log_data: dict, step: int) -> None:
        return

    def save_data(
        self,
        epoch: int,
        env_step: int,
        update_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        if self.save_interval and (env_step - self.last_save_step) >= self.save_interval:
            if save_checkpoint_fn:
                save_checkpoint_fn(epoch, env_step, update_step)
            self.last_save_step = env_step

    def restore_data(self) -> tuple[int, int, int]:
        return 0, 0, 0

    def get_checkpoint_dir(self) -> str:
        return self.ckpt_dir

    def get_wandb_run_id(self) -> Optional[str]:
        if self._wandb_run is None:
            return None
        return getattr(self._wandb_run, "id", None)

    def is_resume(self) -> bool:
        return self._is_resume

    @staticmethod
    def restore_logged_data(log_path: str) -> dict:
        return {}

    def finalize(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
        if self._csv_file is not None:
            self._csv_file.close()
