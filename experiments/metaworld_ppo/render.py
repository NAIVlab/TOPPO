#!/usr/bin/env python3
"""Render multi-task RL agent videos via a fork-based task-parallel pool."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import imageio
import numpy as np
import torch

from mtrl.utils.config import load_config
from mtrl.envs.metaworld_wrapper import TaskIDWrapper
from mtrl.models.networks import MultiTaskActor, SharedPreprocessNet


DEFAULT_CAMERAS = [
    "corner",
    "corner2",
    "corner3",
    "topview",
    "behindGripper",
    "gripperPOV",
]


# Populated in the parent before fork; children read via CoW.
_W: Dict[str, Any] = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Render MTRL Agent")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the experiment config YAML; inferred from --checkpoint if omitted.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default="videos",
                        help="Directory to save rendered videos")
    parser.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS,
                        help="List of camera names to render")
    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Number of episodes to render per task/camera")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Pool size (default: min(num_jobs, cpu_count())).")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Alias of --num-workers, kept for backwards compatibility.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed; episode i uses seed+i.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video frame rate.")
    return parser.parse_args()


def _build_actor(config, task_names: List[str], obs_dim: int, act_dim: int) -> MultiTaskActor:
    preprocess_net = SharedPreprocessNet(
        input_dim=obs_dim,
        hidden_sizes=config.network.preprocess_net,
        activation=config.network.activation,
    )
    actor = MultiTaskActor(
        action_shape=act_dim,
        task_names=task_names,
        preprocess_net=preprocess_net,
        actor_hidden_sizes=config.network.actor,
        actor_head_hidden_sizes=config.network.actor_head,
        activation=config.network.activation,
        max_action=1.0,
        unbounded=True,
        conditioned_sigma=False,
    )
    return actor


def _load_actor_from_ckpt(config, task_names, obs_dim, act_dim, ckpt_path: str) -> MultiTaskActor:
    actor = _build_actor(config, task_names, obs_dim, act_dim)

    # weights_only=False — checkpoints carry non-tensor state (configs, step counters).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("algorithm_state_dict", ckpt)

    current = actor.state_dict()
    new_sd = {}
    prefixes = ["policy.actor.", "actor.", "algorithm.policy.actor."]
    matched = 0
    for key, value in state_dict.items():
        if key in current:
            new_sd[key] = value
            matched += 1
            continue
        for p in prefixes:
            if key.startswith(p):
                stripped = key[len(p):]
                if stripped in current:
                    new_sd[stripped] = value
                    matched += 1
                    break
    if matched == 0:
        print("WARNING: No matching keys found in checkpoint!")
    else:
        print(f"Matched {matched} keys.")
    actor.load_state_dict(new_sd, strict=False)
    actor.eval()
    return actor


def _resolve_benchmark(config) -> Tuple[Dict[str, Any], List[Any], List[str]]:
    import metaworld
    if config.env.env_name == "MT10":
        bench = metaworld.MT10(seed=config.seed)
    elif config.env.env_name == "MT25":
        bench = metaworld.MT25(seed=config.seed)
    elif config.env.env_name == "MT50":
        bench = metaworld.MT50(seed=config.seed)
    else:
        raise ValueError(f"Unknown benchmark: {config.env.env_name}")
    return bench.train_classes, bench.train_tasks, sorted(bench.train_classes.keys())


def _populate_worker_globals(actor, train_classes, train_tasks, task_names,
                             task_name_to_id, num_tasks, config, max_steps):
    _W["actor"] = actor
    _W["train_classes"] = train_classes
    _W["train_tasks"] = train_tasks
    _W["task_names"] = task_names
    _W["task_name_to_id"] = task_name_to_id
    _W["num_tasks"] = num_tasks
    _W["config"] = config
    _W["max_steps"] = max_steps


def _worker_run_job(job: Dict[str, Any]) -> Tuple[str, int, float]:
    # Avoid N-workers × N-threads CPU thrashing; the MLP forward is single-batch.
    torch.set_num_threads(1)

    config = _W["config"]
    task_name = job["task"]
    task_cls = _W["train_classes"][task_name]

    env_task = None
    for t in _W["train_tasks"]:
        if t.env_name == task_name:
            env_task = t
            break
    if env_task is None:
        raise RuntimeError(f"No task spec found for {task_name}")

    if job["camera"]:
        env = task_cls(render_mode="rgb_array", camera_name=job["camera"])
    else:
        env = task_cls(render_mode="rgb_array")
    env.set_task(env_task)

    if config.env.max_episode_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=config.env.max_episode_steps)

    if config.env.use_one_hot:
        env = TaskIDWrapper(
            env,
            task_id=_W["task_name_to_id"][task_name],
            num_tasks=_W["num_tasks"],
            task_name=task_name,
        )

    reset_out = env.reset(seed=int(job["seed"]))
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    actor = _W["actor"]
    max_steps = _W["max_steps"]
    frames = []

    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            (mu, _sigma), _ = actor(obs_t)
            # Match training/eval policy: unbounded + clip to [-1, 1].
            action = mu.clamp(-1.0, 1.0).squeeze(0).numpy()

            step_out = env.step(action)
            if len(step_out) == 5:
                obs, _r, term, trunc, _info = step_out
                done = bool(term) or bool(trunc)
            else:
                obs, _r, done, _info = step_out

            frame = env.render()
            if frame is not None:
                frames.append(np.flipud(frame))
            if done:
                break

    elapsed = time.time() - t0
    env.close()

    if frames:
        imageio.mimsave(str(job["filename"]), frames, fps=int(job["fps"]))

    return str(job["filename"]), len(frames), elapsed


def main():
    args = parse_args()

    if args.config is None:
        ckpt_path = Path(args.checkpoint).resolve()
        possible = ckpt_path.parent.parent / "config.yaml"
        if possible.exists():
            args.config = str(possible)
            print(f"Inferred config path: {args.config}")
        else:
            raise ValueError(f"Could not infer config from {args.checkpoint}; pass --config.")

    # load_config() reads sys.argv (tyro). Hand it just `--config <path>` so
    # render-only flags don't confuse it.
    sys.argv = [sys.argv[0], "--config", args.config]
    config = load_config()

    # Build benchmark in parent (NOT in each worker — that's seconds × 50 = minutes).
    print(f"Building {config.env.env_name} benchmark in parent...")
    train_classes, train_tasks, task_names = _resolve_benchmark(config)
    task_name_to_id = {n: i for i, n in enumerate(task_names)}
    num_tasks = len(task_names)
    print(f"Resolved {num_tasks} tasks.")

    sample = train_classes[task_names[0]](render_mode=None)
    obs_dim_base = int(np.prod(sample.observation_space.shape))
    act_dim = int(np.prod(sample.action_space.shape))
    sample.close()
    obs_dim = obs_dim_base + num_tasks if config.env.use_one_hot else obs_dim_base
    print(f"Obs Dim: {obs_dim}, Act Dim: {act_dim}")

    print(f"Loading checkpoint from {args.checkpoint}")
    actor = _load_actor_from_ckpt(config, task_names, obs_dim, act_dim, args.checkpoint)

    output_dir = Path(args.output_dir)
    ckpt_path = Path(args.checkpoint).resolve()
    save_dir = output_dir / ckpt_path.parent.parent.name / ckpt_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering to {save_dir}")

    max_steps = config.env.max_episode_steps if config.env.max_episode_steps > 0 else 500
    jobs: List[Dict[str, Any]] = []
    for tname in task_names:
        for cam in args.cameras:
            for ep in range(args.num_episodes):
                seed = (args.seed + ep) if args.seed is not None else int(np.random.randint(0, 100000))
                jobs.append({
                    "task": tname,
                    "camera": cam,
                    "episode": ep,
                    "seed": seed,
                    "filename": save_dir / f"{tname}_{cam}_ep{ep}.mp4",
                    "fps": args.fps,
                })
    total_jobs = len(jobs)
    print(f"Total render jobs: {total_jobs}")

    requested = args.num_workers if args.num_workers is not None else args.num_envs
    if requested is None:
        num_workers = min(total_jobs, os.cpu_count() or 1)
    else:
        num_workers = min(requested, total_jobs)
    print(f"Pool size: {num_workers}")

    _populate_worker_globals(
        actor, train_classes, train_tasks, task_names,
        task_name_to_id, num_tasks, config, max_steps,
    )

    # fork (not spawn) — parent is CPU-only, so CoW hands children the loaded
    # actor + benchmark without re-importing or re-constructing anything.
    ctx = mp.get_context("fork")

    t_start = time.time()
    completed = 0
    with ctx.Pool(num_workers) as pool:
        for filename, n_frames, elapsed in pool.imap_unordered(_worker_run_job, jobs):
            completed += 1
            tag = f"[{completed}/{total_jobs}]"
            if n_frames:
                print(f"{tag} saved {filename} ({n_frames} frames, {elapsed:.1f}s)")
            else:
                print(f"{tag} no frames captured for {filename}")
    print(f"All {total_jobs} jobs done in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
