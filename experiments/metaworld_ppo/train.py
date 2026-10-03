#!/usr/bin/env python3
"""Multi-task PPO training entry point (MetaWorld MT10 / MT25 / MT50).

Per-side combiners (pcgrad / cagrad / fairgrad) are selected via
``config.gradient.actor_method`` / ``critic_method``.

Usage::

    python train.py --config configs/ppo_mt10_default.yaml
    python train.py --config configs/ppo_mt10_fairgrad.yaml

All hyperparameters can be overridden on the CLI (Tyro).
"""

from __future__ import annotations

import os
import random
import signal
import sys
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from torch.distributions import Independent, Normal

from mtrl.utils.config import Config, load_config, save_config
from mtrl.utils.logger import MTRLLogger
from mtrl.utils.checkpoints import CheckpointManager


# SIGUSR1 → flip the flag → training loop checkpoints + exits at the
# next epoch boundary (graceful shutdown for walltime-bounded runs).
_GRACEFUL_SHUTDOWN_REQUESTED = False


def _install_graceful_shutdown_handler() -> None:
    def _handler(signum: int, frame: Any) -> None:
        global _GRACEFUL_SHUTDOWN_REQUESTED
        if _GRACEFUL_SHUTDOWN_REQUESTED:
            return
        _GRACEFUL_SHUTDOWN_REQUESTED = True
        print(
            f"[Train] Received signal {signum} — will checkpoint and exit "
            "at next epoch boundary."
        )

    signal.signal(signal.SIGUSR1, _handler)
from mtrl.envs.metaworld_wrapper import make_metaworld_envs
from mtrl.models.networks import (
    SharedPreprocessNet,
    MultiTaskCritic,
    MultiTaskActor,
)
from mtrl.models.multitask_policy import MultiTaskProbabilisticActorPolicy
from mtrl.algos.multitask_ppo import MultiTaskPPO


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _dist_fn(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Independent:
    loc, scale = loc_scale
    return Independent(Normal(loc=loc, scale=scale), reinterpreted_batch_ndims=1)


def create_ppo_algorithm(
    config: Config,
    task_names: list[str],
    obs_dim: int,
    act_dim: int,
    action_space: Any,
    observation_space: Any,
    device: str,
) -> MultiTaskPPO:
    """Build a configured :class:`MultiTaskPPO` instance."""
    from tianshou.algorithm.optim import AdamOptimizerFactory
    from mtrl.models.linear_baseline import LinearFeatureBaseline

    use_linear_baseline = config.use_linear_baseline
    if not use_linear_baseline:
        # Auto-detect: empty critic/actor_head/critic_head ⇒ linear baseline.
        auto_detected = (
            config.network.critic is not None and len(config.network.critic) == 0
            and config.network.actor_head is not None and len(config.network.actor_head) == 0
            and config.network.critic_head is not None and len(config.network.critic_head) == 0
        )
        if auto_detected:
            import warnings
            warnings.warn(
                "Linear baseline auto-detected from empty critic/actor_head/critic_head. "
                "Consider setting use_linear_baseline=true explicitly in your config.",
                UserWarning,
                stacklevel=2,
            )
            use_linear_baseline = True

    if use_linear_baseline:
        print("[Train] Using Linear Feature Baseline (no neural network critic)")

    if config.ppo.use_popart and use_linear_baseline:
        raise ValueError(
            "ppo.use_popart=true is incompatible with use_linear_baseline=true: "
            "PopArt requires a renormalizable neural critic head.  Pick one."
        )

    use_hlgauss = config.ppo.critic_loss_mode == "hlgauss"
    if use_hlgauss:
        if config.ppo.hlgauss_vmin is None or config.ppo.hlgauss_vmax is None:
            raise ValueError(
                "ppo.critic_loss_mode='hlgauss' requires explicit "
                "ppo.hlgauss_vmin / ppo.hlgauss_vmax (the value support must "
                "be chosen per benchmark/reward version)."
            )
        if use_linear_baseline:
            raise ValueError(
                "ppo.critic_loss_mode='hlgauss' requires a neural critic "
                "(use_linear_baseline must be false)."
            )
    if config.ppo.use_return_scaling and use_linear_baseline:
        raise ValueError(
            "ppo.use_return_scaling=true requires a neural critic "
            "(use_linear_baseline must be false)."
        )
    # Reward normalization cannot be combined with PopArt or return scaling:
    # each of those also rescales the value targets per task, so the pair
    # would normalize twice.  Reward normalization + HL-Gauss is allowed.
    # Reward normalization rescales the reward inside the env, hence
    # minibatch.returns -- the very quantity HL-Gauss projects onto its fixed
    # support -- so the pair is a genuine composition rather than a double
    # normalization, and it tests whether bringing per-task targets into a
    # shared range makes a single fixed HL-Gauss support viable.  HL-Gauss
    # also composes with return scaling (the target is divided by s_i) and
    # with PopArt (ART: PopArt supplies only the target statistics); the
    # remaining pair, PopArt + return scaling, is rejected in
    # MultiTaskPPO.__init__.
    if config.env.reward_normalization_method is not None and (
        config.ppo.use_popart or config.ppo.use_return_scaling
    ):
        raise ValueError(
            "env.reward_normalization_method="
            f"{config.env.reward_normalization_method!r} cannot be combined "
            "with use_popart / use_return_scaling: both would normalize the "
            "value targets a second time.  Reward normalization combines "
            "only with critic_loss_mode='hlgauss'."
        )

    preprocess_net = SharedPreprocessNet(
        input_dim=obs_dim,
        hidden_sizes=config.network.preprocess_net,
        activation=config.network.activation,
        use_layer_norm=config.network.ln_preprocess,
    )

    actor = MultiTaskActor(
        action_shape=act_dim,
        task_names=task_names,
        preprocess_net=preprocess_net,
        actor_hidden_sizes=config.network.actor,
        actor_head_hidden_sizes=config.network.actor_head,
        activation=config.network.activation,
        max_action=1.0,
        unbounded=True,  # PPO: unbounded action then clip downstream.
        conditioned_sigma=False,
        body_use_layer_norm=config.network.ln_actor,
        head_use_layer_norm=config.network.ln_actor_head,
    ).to(device)

    if use_linear_baseline:
        # Dummy critic only used to satisfy Tianshou's constructor.
        critic = torch.nn.Linear(obs_dim, 1).to(device)
        raw_obs_dim = obs_dim - len(task_names)
        linear_baseline = LinearFeatureBaseline(
            obs_dim=raw_obs_dim,
            num_tasks=len(task_names),
        )
    else:
        critic = MultiTaskCritic(
            task_names=task_names,
            preprocess_net=preprocess_net,
            critic_hidden_sizes=config.network.critic,
            critic_head_hidden_sizes=config.network.critic_head,
            activation=config.network.activation,
            act_dim=0,
            popart_enabled=config.ppo.use_popart,
            popart_min_std=config.ppo.popart_min_std,
            popart_method=config.ppo.popart_method,
            popart_ema_decay=config.ppo.popart_ema_decay,
            body_use_layer_norm=config.network.ln_critic,
            head_use_layer_norm=config.network.ln_critic_head,
            hlgauss_num_bins=config.ppo.hlgauss_num_bins if use_hlgauss else None,
            hlgauss_vmin=config.ppo.hlgauss_vmin,
            hlgauss_vmax=config.ppo.hlgauss_vmax,
            hlgauss_sigma_ratio=config.ppo.hlgauss_sigma_ratio,
        ).to(device)
        linear_baseline = None

    policy = MultiTaskProbabilisticActorPolicy(
        actor=actor,
        dist_fn=_dist_fn,
        task_names=task_names,
        deterministic_eval=True,
        action_space=action_space,
        observation_space=observation_space,
        action_scaling=True,
        action_bound_method="clip",
    ).to(device)

    optim_factory = AdamOptimizerFactory(lr=config.training.learning_rate)

    common_kwargs = dict(
        policy=policy,
        critic=critic,
        optim=optim_factory,
        task_names=task_names,
        eps_clip=config.ppo.clip_range,
        value_clip=config.ppo.value_clip,
        vf_coef=config.ppo.vf_coef,
        ent_coef=config.ppo.ent_coef,
        max_grad_norm=config.training.max_grad_norm,
        gae_lambda=config.ppo.gae_lambda,
        max_batchsize=config.training.batch_size,
        minibatch_sampling=config.training.minibatch_sampling,
        gamma=config.training.gamma,
        advantage_normalization=config.ppo.advantage_normalization,
        recompute_advantage=config.ppo.recompute_advantage,
        use_linear_baseline=use_linear_baseline,
        linear_baseline=linear_baseline,
        use_popart=config.ppo.use_popart,
        use_return_scaling=config.ppo.use_return_scaling,
        return_scaling_min_sigma=config.ppo.return_scaling_min_sigma,
        return_scaling_truncation_as_terminal=config.ppo.return_scaling_truncation_as_terminal,
        critic_loss_mode=config.ppo.critic_loss_mode,
    )
    algorithm = MultiTaskPPO(
        **common_kwargs,
        actor_method=config.gradient.actor_method,
        critic_method=config.gradient.critic_method,
        cagrad_alpha=config.gradient.cagrad_alpha,
        fairgrad_actor_alpha=config.gradient.fairgrad_actor_alpha,
        fairgrad_critic_alpha=config.gradient.fairgrad_critic_alpha,
        # Per-task gradient snapshot dump.  Inert when dump_grads_dir is
        # None; ``run_training`` has already resolved the magic value
        # "auto" to ``<exp_dir>/grad_snapshots`` by the time we get here.
        dump_grads_dir=config.logging.dump_grads_dir,
        dump_grads_every_collects=config.logging.dump_grads_every_collects,
    )
    return algorithm



def make_episode_hook(algorithm: Any, mode: str = "train") -> Any:
    """Per-task stats hook fired on episode completion."""
    def hook(episode_batch: Any) -> dict[str, Any] | None:
        infos = episode_batch.info

        # The train-side MultiTaskCollector patches cross-collect cumulative
        # reward/win/steps onto the batch before calling this hook; the
        # plain Collector used for tests does not, so we fall back to
        # buffer-summed values below.
        cumulative_reward = getattr(episode_batch, "_cumulative_reward", None)
        cumulative_win = getattr(episode_batch, "_cumulative_win", None)
        cumulative_steps = getattr(episode_batch, "_cumulative_steps", None)

        if cumulative_reward is not None:
            episode_reward = float(cumulative_reward)
        else:
            # Sum of buffer rewards, preferring unnormalized raw_reward when
            # available.
            episode_reward = float(np.sum(episode_batch.rew))
            try:
                if hasattr(infos, "raw_reward"):
                    episode_reward = float(np.sum(infos["raw_reward"]))
                elif hasattr(infos, '__len__') and len(infos) > 0:
                    if isinstance(infos[0], dict) and "raw_reward" in infos[0]:
                        episode_reward = float(sum(
                            info.get("raw_reward", 0.0) for info in infos
                        ))
            except (TypeError, KeyError, AttributeError):
                pass

        last_info = infos[-1]
        task_name = last_info.get("task_name", "unknown")

        if cumulative_win is not None:
            win = float(cumulative_win)
        else:
            # MetaWorld convention: win is success at ANY step in the episode.
            try:
                if hasattr(infos, "win"):
                    win = float(np.max(infos["win"]))
                elif hasattr(infos, '__len__') and len(infos) > 0:
                    wins = [
                        float(info.get("win", 0.0))
                        if isinstance(info, dict) else 0.0
                        for info in infos
                    ]
                    win = max(wins)
                else:
                    win = float(last_info.get("win", 0.0))
            except (TypeError, KeyError):
                win = float(last_info.get("win", 0.0))

        length = int(cumulative_steps) if cumulative_steps is not None else len(episode_batch)

        algorithm.update_per_task_stats(
            task_name=task_name,
            reward=episode_reward,
            win=win,
            length=length,
            mode=mode,
        )
        return None
    return hook


def _named_trainable_parameters(module: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, param)
        for name, param in module.named_parameters()
        if param.requires_grad
    ]


def _direct_trainable_parameters(module: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, param)
        for name, param in module.named_parameters(recurse=False)
        if param.requires_grad
    ]


def _format_int(value: int) -> str:
    return f"{value:,}"


def _print_parameter_inventory(algorithm: Any, algo_name: str) -> None:
    """Print a parameter inventory; counts shared modules only once."""
    policy = getattr(algorithm, "policy", None)
    actor = getattr(policy, "actor", None)
    critic = getattr(algorithm, "critic", None)
    critic2 = getattr(algorithm, "critic2", None)

    component_specs: list[tuple[str, list[tuple[str, torch.nn.Parameter]]]] = []

    def add_module_components(prefix: str, module: Any) -> None:
        if not isinstance(module, torch.nn.Module):
            return

        known_component = False
        preprocess = getattr(module, "preprocess", None)
        if isinstance(preprocess, torch.nn.Module):
            component_specs.append((f"{prefix}.preprocess", _named_trainable_parameters(preprocess)))
            known_component = True

        actor_body = getattr(module, "actor_body", None)
        if isinstance(actor_body, torch.nn.Module):
            component_specs.append((f"{prefix}.actor_body", _named_trainable_parameters(actor_body)))
            known_component = True

        actor_output = getattr(module, "actor_output", None)
        if isinstance(actor_output, torch.nn.Module):
            component_specs.append((f"{prefix}.actor_output", _named_trainable_parameters(actor_output)))
            known_component = True

        mu_heads = getattr(module, "mu_heads", None)
        if isinstance(mu_heads, torch.nn.Module):
            component_specs.append((f"{prefix}.mu_heads", _named_trainable_parameters(mu_heads)))
            known_component = True

        sigma_heads = getattr(module, "sigma_heads", None)
        if isinstance(sigma_heads, torch.nn.Module):
            component_specs.append((f"{prefix}.sigma_heads", _named_trainable_parameters(sigma_heads)))
            known_component = True

        critic_body = getattr(module, "critic_body", None)
        if isinstance(critic_body, torch.nn.Module):
            component_specs.append((f"{prefix}.critic_body", _named_trainable_parameters(critic_body)))
            known_component = True

        value_heads = getattr(module, "value_heads", None)
        if isinstance(value_heads, torch.nn.Module):
            component_specs.append((f"{prefix}.value_heads", _named_trainable_parameters(value_heads)))
            known_component = True

        direct_params = _direct_trainable_parameters(module)
        if direct_params:
            component_specs.append((f"{prefix}.direct_parameters", direct_params))
            known_component = True

        if not known_component:
            component_specs.append((prefix, _named_trainable_parameters(module)))

    add_module_components("actor", actor)
    add_module_components("critic", critic)
    add_module_components("critic2", critic2)

    linear_baseline = getattr(algorithm, "_linear_baseline", None)

    seen_param_ids: set[int] = set()
    unique_total = 0
    unique_tensor_total = 0

    print(f"[Train] Parameter inventory for {algo_name.upper()}:")
    for label, named_params in component_specs:
        if not named_params:
            continue

        component_param_count = sum(param.numel() for _, param in named_params)
        component_tensor_count = len(named_params)

        new_named_params = [
            (name, param)
            for name, param in named_params
            if id(param) not in seen_param_ids
        ]
        new_param_count = sum(param.numel() for _, param in new_named_params)
        new_tensor_count = len(new_named_params)

        for _, param in new_named_params:
            seen_param_ids.add(id(param))

        unique_total += new_param_count
        unique_tensor_total += new_tensor_count

        status = "counted"
        if new_tensor_count == 0:
            status = "shared reference"
        elif new_tensor_count < component_tensor_count:
            status = "partially shared"

        print(
            "[Train]   "
            f"{label}: {_format_int(component_param_count)} parameters "
            f"across {_format_int(component_tensor_count)} tensors "
            f"({status}; unique contribution {_format_int(new_param_count)} "
            f"parameters across {_format_int(new_tensor_count)} tensors)"
        )

    if linear_baseline is not None:
        print("[Train]   linear_baseline: non-PyTorch baseline object (0 trainable torch parameters)")

    print(
        "[Train] Total unique trainable parameters: "
        f"{_format_int(unique_total)} across {_format_int(unique_tensor_total)} tensors"
    )


def run_training(config: Config) -> None:
    print(f"[Train] Config: algo={config.algo}, env={config.env.env_name}, "
          f"gradient=actor:{config.gradient.actor_method}/critic:{config.gradient.critic_method}")

    set_seed(config.seed)

    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Train] CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"[Train] Using device: {device}")

    if not config.env.use_one_hot:
        raise ValueError(
            "use_one_hot=False is not supported: the multi-task actor/critic "
            "architecture requires one-hot task encoding in observations. "
            "Set env.use_one_hot=true."
        )

    print(f"[Train] Creating {config.env.env_name} environments...")
    train_envs, test_envs, task_name_to_id, task_names, num_goals_per_task, task_env_map = make_metaworld_envs(
        config, seed=config.seed
    )

    # Init logging only after vector env workers are forked: ShmemVectorEnv
    # uses Linux fork() and starting TB / W&B threads first can deadlock
    # child env creation.
    logger = MTRLLogger(
        config=config,
        use_wandb=config.logging.use_wandb,
        use_tensorboard=config.logging.use_tensorboard,
        use_csv=config.logging.use_csv,
        train_interval=config.logging.train_interval,
        # Don't throttle test logs — test frequency is gated at the epoch
        # level in the loop below.
        test_interval=1,
        update_interval=config.logging.update_interval,
        info_interval=config.logging.info_interval,
        save_interval=config.logging.save_interval,
    )
    print(f"[Train] Experiment directory: {logger.exp_dir}")

    # Resolve the magic dump_grads_dir value "auto" -> ``<exp_dir>/grad_snapshots``.
    # Mutated in place BEFORE ``create_ppo_algorithm`` reads it; any other
    # absolute / relative path is passed through verbatim.
    if config.logging.dump_grads_dir == "auto":
        config.logging.dump_grads_dir = os.path.join(logger.exp_dir, "grad_snapshots")
    if config.logging.dump_grads_dir:
        os.makedirs(config.logging.dump_grads_dir, exist_ok=True)
        print(f"[Train] Per-task gradient dump dir: {config.logging.dump_grads_dir} "
              f"(every {config.logging.dump_grads_every_collects} collects)")

    # Eval protocol: one episode per (goal, task).
    num_test_episodes = num_goals_per_task * len(task_names)
    print(f"[Train] Eval: {num_goals_per_task} goals/task × {len(task_names)} tasks = {num_test_episodes} test episodes")

    obs_space = train_envs.observation_space[0] if hasattr(train_envs.observation_space, '__getitem__') else train_envs.observation_space
    act_space = train_envs.action_space[0] if hasattr(train_envs.action_space, '__getitem__') else train_envs.action_space

    if hasattr(obs_space, 'shape'):
        obs_dim = int(np.prod(obs_space.shape))
    else:
        obs_dim = obs_space.n

    if hasattr(act_space, 'shape'):
        act_dim = int(np.prod(act_space.shape))
    else:
        act_dim = act_space.n

    print(f"[Train] obs_dim={obs_dim}, act_dim={act_dim}, num_tasks={len(task_names)}")
    print(f"[Train] Tasks: {task_names}")

    print(f"[Train] Creating {config.algo.upper()} algorithm...")
    algorithm = create_ppo_algorithm(
        config=config,
        task_names=task_names,
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_space=act_space,
        observation_space=obs_space,
        device=device,
    )

    _print_parameter_inventory(algorithm, config.algo)

    from tianshou.data.collector import Collector
    from tianshou.data.buffer.vecbuf import VectorReplayBuffer

    train_buffer = VectorReplayBuffer(
        total_size=config.training.step_per_epoch,
        buffer_num=config.env.num_envs,
    )

    episode_hook = make_episode_hook(algorithm, mode="train")

    test_collector = None
    if test_envs is not None and config.logging.test_interval > 0:
        test_collector = Collector(
            policy=algorithm,
            env=test_envs,
            on_episode_done_hook=make_episode_hook(algorithm, mode="test"),
        )

    logger.set_algorithm(algorithm)

    ckpt_manager = CheckpointManager(
        ckpt_dir=logger.get_checkpoint_dir(),
        config=config,
        save_best=config.logging.save_best,
        checkpoint_interval=config.logging.save_interval,
    )

    steps_per_collect = config.training.step_per_collect
    steps_per_epoch = config.training.step_per_epoch
    max_epochs = config.training.epoch

    from mtrl.utils.multitask_collector import MultiTaskCollector
    train_collector = MultiTaskCollector(
        policy=algorithm,
        env=train_envs,
        buffer=train_buffer,
        exploration_noise=True,
        on_episode_done_hook=episode_hook,
    )

    from tianshou.trainer import OnPolicyTrainer, OnPolicyTrainerParams

    # baselines_return reward-normalizer statistics live inside the env
    # worker processes; snapshot them onto the algorithm right before every
    # rolling save so preemption resume doesn't silently reset the per-task
    # reward scale (they ride along in _PERSISTABLE_EXTRA_ATTRS).
    _uses_env_reward_norm = (
        config.env.reward_normalization_method == "baselines_return"
    )

    def _capture_reward_norm_state() -> None:
        if _uses_env_reward_norm:
            algorithm._reward_norm_env_states = train_envs.get_env_attr(
                "reward_norm_state"
            )

    # ``trainer`` is defined below; save_checkpoint_fn closes over its name
    # so resume state (counters + wandb id) ends up in every rolling save.
    def save_checkpoint_fn(epoch, env_step, update_step):
        if ckpt_manager.should_save(env_step):
            _capture_reward_norm_state()
            path = ckpt_manager.save_checkpoint(
                algorithm=algorithm,
                step=env_step,
                epoch=epoch,
                trainer=trainer,
                wandb_run_id=logger.get_wandb_run_id(),
            )
            return path
        return ""

    # Test frequency is gated at epoch level by config.logging.test_interval;
    # the trainer's built-in test schedule is disabled here.
    trainer_params = OnPolicyTrainerParams(
        max_epochs=max_epochs,
        epoch_num_steps=steps_per_epoch,
        training_collector=train_collector,
        collection_step_num_env_steps=steps_per_collect,
        batch_size=config.training.batch_size,
        update_step_num_repetitions=config.training.repeat_per_collect,
        test_collector=None,
        test_step_num_episodes=0,
        test_in_training=False,
        stop_fn=None,
        test_fn=None,
        save_best_fn=None,  # Handled manually in the epoch loop below.
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        verbose=True,
        show_progress=True,
    )

    trainer = OnPolicyTrainer(algorithm=algorithm, params=trainer_params)
    test_interval = config.logging.test_interval
    do_test = test_collector is not None and test_interval > 0

    # Restore AFTER trainer.reset() so we overwrite its zeroed counters with
    # the saved values; failures here are fatal so we don't silently
    # cold-start when the user expected to resume.
    trainer.reset(reset_collectors=True, reset_collector_buffers=False)

    resumed_epoch = 0
    latest_ckpt_path = CheckpointManager.find_latest(logger.get_checkpoint_dir())
    if latest_ckpt_path is not None:
        print(f"[Train] Resuming from checkpoint: {latest_ckpt_path}")
        checkpoint = CheckpointManager.load_checkpoint(latest_ckpt_path, device=device)
        # HL-Gauss support/centers are rebuilt from config (persistent=False),
        # so a resume with a changed support would silently reinterpret the
        # trained logits on a different grid.  Refuse it.
        if config.ppo.critic_loss_mode == "hlgauss":
            saved_ppo = (checkpoint.get("config") or {}).get("ppo", {})
            for key in ("critic_loss_mode", "hlgauss_num_bins", "hlgauss_vmin",
                        "hlgauss_vmax", "hlgauss_sigma_ratio"):
                saved = saved_ppo.get(key)
                active = getattr(config.ppo, key)
                if saved is not None and saved != active:
                    raise ValueError(
                        f"Refusing to resume: checkpoint was trained with "
                        f"ppo.{key}={saved!r} but the active config has "
                        f"{active!r}. The HL-Gauss support must not change "
                        f"across a resume."
                    )
        resume_info = ckpt_manager.restore(
            checkpoint=checkpoint,
            algorithm=algorithm,
            trainer=trainer,
        )
        resumed_epoch = int(resume_info["epoch"])
        print(
            f"[Train] Resumed at epoch={resumed_epoch}, "
            f"env_step={trainer._env_step}, best_reward={resume_info['best_reward']:.2f}"
        )
        if _uses_env_reward_norm:
            saved_states = getattr(algorithm, "_reward_norm_env_states", None)
            restored = 0
            for i, st in enumerate(saved_states or []):
                if isinstance(st, dict):
                    # Fresh envs start mid-nothing: zero the episode
                    # accumulator, keep the all-time running statistics.
                    train_envs.set_env_attr(
                        "reward_norm_state", dict(st, ret=0.0), id=i,
                    )
                    restored += 1
            if restored:
                print(
                    f"[Train] Restored baselines_return reward-normalizer "
                    f"stats for {restored} envs"
                )
            else:
                print(
                    "[Train] WARNING: resuming a baselines_return run without "
                    "saved reward-normalizer stats — the reward scale will "
                    "re-warm from scratch."
                )

    _install_graceful_shutdown_handler()

    print(f"[Train] Starting training...")
    print(f"[Train] Epochs: {max_epochs}, Steps/epoch: {steps_per_epoch}")
    if do_test:
        print(f"[Train] Test every {test_interval} epoch(s)")

    # Initial baseline test (skip on resume — history is already there).
    if do_test and resumed_epoch == 0:
        algorithm.eval()
        test_collector.reset()
        test_collector.collect(n_episode=num_test_episodes)
        test_stats = algorithm.get_per_task_test_stats()
        rewards = [
            m["reward"] for m in test_stats.values()
            if "reward" in m and m.get("num_episodes", 0) > 0
        ]
        logger.log_test_data({}, 0)
        if rewards:
            mean_reward = float(np.mean(rewards))
            print(f"[Train] Initial test (step 0): mean_reward={mean_reward:.2f}")
        algorithm.train()

    start_time = time.time()
    best_reward = float(ckpt_manager.best_reward)

    start_epoch = resumed_epoch + 1
    completed_epoch = resumed_epoch
    graceful_exit = False

    for epoch in range(start_epoch, max_epochs + 1):
        # Toggle test_collector on test epochs only.
        if do_test and epoch % test_interval == 0:
            trainer.params.test_collector = test_collector
            trainer.params.test_step_num_episodes = num_test_episodes
        else:
            trainer.params.test_collector = None
            trainer.params.test_step_num_episodes = 0

        epoch_stats = trainer.execute_epoch()
        completed_epoch = epoch

        if epoch_stats.test_collect_stat is not None:
            test_stats = algorithm.get_per_task_test_stats()
            rewards = [
                m["reward"] for m in test_stats.values()
                if "reward" in m and m.get("num_episodes", 0) > 0
            ]
            if rewards:
                mean_reward = float(np.mean(rewards))
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    ckpt_manager.save_best_if_improved(
                        algorithm=algorithm,
                        step=epoch * steps_per_epoch,
                        mean_reward=mean_reward,
                        epoch=epoch,
                        trainer=trainer,
                        wandb_run_id=logger.get_wandb_run_id(),
                    )

        if _GRACEFUL_SHUTDOWN_REQUESTED:
            graceful_exit = True
            break

    elapsed = time.time() - start_time

    if graceful_exit:
        print(
            f"\n[Train] Graceful shutdown at epoch {completed_epoch} "
            f"(elapsed {elapsed:.1f}s).  Saving rolling checkpoint and exiting."
        )
        _capture_reward_norm_state()
        ckpt_manager.save_checkpoint(
            algorithm=algorithm,
            step=completed_epoch * steps_per_epoch,
            epoch=completed_epoch,
            trainer=trainer,
            wandb_run_id=logger.get_wandb_run_id(),
        )
        logger.finalize()
        train_envs.close()
        if test_envs is not None:
            test_envs.close()
        print(f"[Train] Rolling checkpoint saved to {logger.get_checkpoint_dir()}")
        sys.exit(0)

    print(f"\n[Train] Training complete in {elapsed:.1f}s")
    if best_reward > -float("inf"):
        print(f"[Train] Best reward: {best_reward:.2f}")

    final_step = max_epochs * steps_per_epoch
    _capture_reward_norm_state()
    ckpt_manager.save_checkpoint(
        algorithm=algorithm,
        step=final_step,
        epoch=max_epochs,
        trainer=trainer,
        wandb_run_id=logger.get_wandb_run_id(),
    )
    print(f"[Train] Final checkpoint saved to {logger.get_checkpoint_dir()}")

    logger.finalize()
    train_envs.close()
    if test_envs is not None:
        test_envs.close()

    print("[Train] Done!")


def main() -> None:
    """Main training entry point."""
    config = load_config()
    run_training(config)


if __name__ == "__main__":
    main()
