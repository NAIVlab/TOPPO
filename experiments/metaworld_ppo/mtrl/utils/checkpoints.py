"""Checkpoint save / load with full resume support.

Layout (written to ``<exp_dir>/checkpoints/``):
  latest.pth     — rolling checkpoint, always overwritten (the resume target)
  best_model.pth — best-reward checkpoint when ``save_best=True``

``latest.pth`` bundles algorithm + optim + algo_extra (Python-dict state)
+ trainer counters + RNG + epoch + wandb run id, so the run can resume
bit-identically on restart.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from mtrl.utils.config import Config


LATEST_FILENAME = "latest.pth"
BEST_FILENAME = "best_model.pth"


def capture_rng_state() -> Dict[str, Any]:
    """Snapshot global RNG state across python/numpy/torch/cuda."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state from a snapshot.

    RNG ByteTensors must be on CPU before the torch setters accept them;
    torch.load with ``map_location="cuda"`` moves them to GPU, so we
    coerce them back here.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_cpu = state["torch_cpu"]
    if isinstance(torch_cpu, torch.Tensor) and torch_cpu.device.type != "cpu":
        torch_cpu = torch_cpu.cpu()
    torch.set_rng_state(torch_cpu)
    cuda = state.get("torch_cuda")
    if cuda is not None and torch.cuda.is_available():
        cuda = [
            t.cpu() if isinstance(t, torch.Tensor) and t.device.type != "cpu" else t
            for t in cuda
        ]
        torch.cuda.set_rng_state_all(cuda)


class CheckpointManager:
    """Save / load training checkpoints with best-reward tracking."""

    def __init__(
        self,
        ckpt_dir: str,
        config: Config,
        save_best: bool = True,
        checkpoint_interval: int = 50000,
    ) -> None:
        self.ckpt_dir = ckpt_dir
        self.config = config
        self.save_best = save_best
        self.checkpoint_interval = checkpoint_interval
        self.best_reward: float = float("-inf")

        os.makedirs(ckpt_dir, exist_ok=True)

    def _build_state(
        self,
        algorithm: torch.nn.Module,
        step: int,
        epoch: int = 0,
        trainer: Optional[Any] = None,
        wandb_run_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Build a full checkpoint state dict.

        Fail-fast on any missing critical state (optim / algo_extra / rng /
        trainer_state) — a partial checkpoint is worse than none, since
        silent loss of Adam momenta or per-task stats produces a misleading
        "successful resume" that is actually a warm/cold-start hybrid.
        """
        # optim is a plain attribute, not a registered nn.Module, so it's
        # not in algorithm.state_dict().
        optim = getattr(algorithm, "optim", None)
        if optim is None or not hasattr(optim, "state_dict"):
            raise RuntimeError(
                "Checkpoint save: algorithm has no usable optim.state_dict(); "
                "refusing to write a checkpoint without Adam momenta."
            )
        optim_state = optim.state_dict()

        if not hasattr(algorithm, "get_training_extra_state"):
            raise RuntimeError(
                "Checkpoint save: algorithm is missing get_training_extra_state; "
                "refusing to write a checkpoint without per-task state."
            )
        algo_extra = algorithm.get_training_extra_state()

        # Tianshou trainer's private counters have no public getters.
        trainer_state = None
        if trainer is not None:
            trainer_state = {
                "epoch": int(getattr(trainer, "_epoch", 0)),
                "env_step": int(getattr(trainer, "_env_step", 0)),
                "update_step": int(getattr(trainer, "_current_update_step", 0)),
            }

        state = {
            "step": step,
            "epoch": int(epoch),
            "algorithm_state_dict": algorithm.state_dict(),
            "optim_state_dict": optim_state,
            "algo_extra_state": algo_extra,
            "trainer_state": trainer_state,
            "rng_state": capture_rng_state(),
            "config": asdict(self.config),
            "best_reward": self.best_reward,
            "wandb_run_id": wandb_run_id,
        }
        if extra is not None:
            state.update(extra)
        return state

    def save_checkpoint(
        self,
        algorithm: torch.nn.Module,
        step: int,
        epoch: int = 0,
        trainer: Optional[Any] = None,
        wandb_run_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Atomically save ``latest.pth`` (tmp + rename)."""
        state = self._build_state(
            algorithm=algorithm,
            step=step,
            epoch=epoch,
            trainer=trainer,
            wandb_run_id=wandb_run_id,
            extra=extra,
        )
        # Self-heal in case the ckpt dir was wiped mid-run.
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, LATEST_FILENAME)
        tmp_path = path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        return path

    def save_best_if_improved(
        self,
        algorithm: torch.nn.Module,
        step: int,
        mean_reward: float,
        epoch: int = 0,
        trainer: Optional[Any] = None,
        wandb_run_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save ``best_model.pth`` if mean_reward improves; return True if saved."""
        if not self.save_best:
            return False
        if mean_reward <= self.best_reward:
            return False
        self.best_reward = mean_reward
        state = self._build_state(
            algorithm=algorithm,
            step=step,
            epoch=epoch,
            trainer=trainer,
            wandb_run_id=wandb_run_id,
            extra=extra,
        )
        state["best_reward"] = self.best_reward
        os.makedirs(self.ckpt_dir, exist_ok=True)
        best_path = os.path.join(self.ckpt_dir, BEST_FILENAME)
        tmp_path = best_path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, best_path)
        print(f"[Checkpoint] New best model saved (reward={mean_reward:.2f})")
        return True

    def should_save(self, step: int) -> bool:
        if self.checkpoint_interval <= 0:
            return False
        return step > 0 and step % self.checkpoint_interval == 0

    @staticmethod
    def find_latest(ckpt_dir: str) -> Optional[str]:
        path = os.path.join(ckpt_dir, LATEST_FILENAME)
        return path if os.path.exists(path) else None

    @staticmethod
    def load_checkpoint(path: str, device: str = "cpu") -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location=device, weights_only=False)

    def restore(
        self,
        checkpoint: Dict[str, Any],
        algorithm: torch.nn.Module,
        trainer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Restore algorithm + optim + trainer + RNG from a loaded checkpoint.

        Fail-fast on any missing critical key; a half-restored run is worse
        than a fresh one since the user believes they resumed.
        """
        if "algorithm_state_dict" not in checkpoint:
            raise RuntimeError("Checkpoint missing algorithm_state_dict — refusing to resume.")
        algorithm.load_state_dict(checkpoint["algorithm_state_dict"])

        optim_state = checkpoint.get("optim_state_dict")
        optim = getattr(algorithm, "optim", None)
        if optim_state is None:
            raise RuntimeError(
                "Checkpoint missing optim_state_dict — refusing to resume without "
                "Adam momenta.  Delete latest.pth to force a cold start."
            )
        if optim is None or not hasattr(optim, "load_state_dict"):
            raise RuntimeError(
                "Algorithm has no usable optim.load_state_dict at resume time."
            )
        optim.load_state_dict(optim_state)

        algo_extra = checkpoint.get("algo_extra_state")
        if algo_extra is None:
            raise RuntimeError(
                "Checkpoint missing algo_extra_state — refusing to resume without "
                "per-task state.  Delete latest.pth to force a cold start."
            )
        if not hasattr(algorithm, "set_training_extra_state"):
            raise RuntimeError(
                "Algorithm has no set_training_extra_state hook at resume time."
            )
        algorithm.set_training_extra_state(algo_extra)

        if trainer is not None:
            tstate = checkpoint.get("trainer_state") or {}
            if "epoch" in tstate:
                trainer._epoch = int(tstate["epoch"])
            if "env_step" in tstate:
                trainer._env_step = int(tstate["env_step"])
            if "update_step" in tstate:
                trainer._current_update_step = int(tstate["update_step"])

        rng = checkpoint.get("rng_state")
        if rng is None:
            raise RuntimeError(
                "Checkpoint missing rng_state — refusing to resume without RNG "
                "continuity.  Delete latest.pth to force a cold start."
            )
        restore_rng_state(rng)

        self.best_reward = float(checkpoint.get("best_reward", float("-inf")))

        return {
            "epoch": int(checkpoint.get("epoch", 0)),
            "step": int(checkpoint.get("step", 0)),
            "wandb_run_id": checkpoint.get("wandb_run_id"),
            "best_reward": self.best_reward,
        }
