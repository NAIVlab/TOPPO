"""Configuration system using dataclasses, YAML, and Tyro."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional, Union, get_args, get_origin, get_type_hints

import yaml


@dataclass
class EnvConfig:
    env_name: str = "MT10"
    max_episode_steps: int = 500
    use_one_hot: bool = True
    num_tasks: int = 10
    terminate_on_success: bool = False
    task_select: Literal["random", "pseudorandom"] = "random"
    # "baselines_return" is the openai/baselines VecNormalize scheme
    # (issue #538): divide the raw reward by the running std of the
    # per-env discounted-return accumulator, no mean subtraction, clip to
    # ±reward_norm_clip.  One env instance hosts one task, so the running
    # statistics are per-task.  Applied to training envs only — eval envs
    # always report raw rewards.
    reward_normalization_method: Optional[
        Literal["gymnasium", "exponential", "baselines_return"]
    ] = None
    reward_norm_clip: float = 10.0
    num_envs: int = 10
    envs_per_task: int = 1
    # MetaWorld reward function version. v2 (default) is the modern shaped
    # reward; v1 is the original sparser reward used in the 2020 benchmark.
    reward_function_version: Literal["v1", "v2"] = "v2"


@dataclass
class NetworkConfig:
    preprocess_net: Optional[List[int]] = None
    actor: Optional[List[int]] = None
    critic: Optional[List[int]] = None
    actor_head: Optional[List[int]] = None
    critic_head: Optional[List[int]] = None
    activation: str = "relu"
    # Per-MLP LayerNorm toggles. Each flag controls LN on the hidden layers
    # of one MLP (pre-activation: Linear → LN → Act). Output projections
    # (final Linear with no activation) never get LN.
    ln_preprocess: bool = False
    ln_actor: bool = False
    ln_critic: bool = True
    ln_actor_head: bool = False
    ln_critic_head: bool = False


@dataclass
class TrainingConfig:
    # Defaults follow ``configs/ppo_mt10_default.yaml`` so a bare run lands on a
    # sensible MT10 setup; MT50 yaml scales the env-step and batch counts.
    epoch: int = 100
    step_per_epoch: int = 200000
    step_per_collect: int = 100000
    repeat_per_collect: int = 16
    batch_size: int = 3125
    minibatch_sampling: Literal["mixed", "stratified"] = "stratified"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    max_grad_norm: float = 1.0


@dataclass
class PPOConfig:
    gae_lambda: float = 0.97
    clip_range: float = 0.2
    value_clip: bool = True
    vf_coef: float = 1e-3
    ent_coef: float = 5e-3
    advantage_normalization: bool = True
    recompute_advantage: bool = True
    # PopArt (Hessel et al., AAAI 2019): per-task value-target normalization
    # with per-task POP affine renormalization that preserves raw V output
    # across stats updates.
    use_popart: bool = True
    popart_min_std: float = 1e-2
    # ART stats-update rule:
    #   "welford" -- (default) unbounded online mean/var via a Welford merge
    #                of every per-task batch ever seen.  Tracks the cumulative
    #                return distribution; this repo's original behavior and a
    #                known divergence from the original PopArt formulation.
    #   "ema"     -- the original-paper exponential moving average (van
    #                Hasselt et al. 2016 / Hessel et al. 2019): per-task
    #                mu <- (1-b) mu + b G, nu <- (1-b) nu + b G^2,
    #                sigma = sqrt(max(nu - mu^2, 0)).  Recency-weighted and
    #                matches the canonical multi-task PopArt reference.
    # popart_ema_decay is the per-sample beta (NOT per-batch); a task batch of
    # N returns is folded in as N sequential EMA updates via a vectorized
    # closed form.  Default 3e-4 = Hessel et al. 2019 multi-task default.
    popart_method: Literal["welford", "ema"] = "welford"
    popart_ema_decay: float = 3e-4
    # Return-based scaling (Schaul et al. 2021, arXiv:2105.05347), per-task:
    # value-loss residuals are divided by s_i = max(σ_i, σ_batch_i, min_sigma)
    # with σ_i² = Var_i[R] + Var_i[γ]·E_i[G²] from all-time fp64 statistics.
    # Mutually exclusive with use_popart and critic_loss_mode="hlgauss".
    use_return_scaling: bool = False
    return_scaling_min_sigma: float = 1e-2
    # Whether a time-limit truncation counts as a terminal step when
    # accumulating Var[gamma].  Schaul et al. set gamma_t = 0 only at TRUE
    # terminals; MetaWorld episodes always end by truncation with
    # bootstrapping, so under the faithful reading Var[gamma] == 0 exactly
    # and sigma^2 collapses to Var[R] — the method never sees the return
    # scale.  Setting this True instead treats every episode end as terminal
    # (Var[gamma] ~= gamma^2/T, per their App. C.2), so the E[G^2] term
    # engages.  Kept as an explicit axis so the baseline is reported under
    # both readings rather than one implementation choice.
    return_scaling_truncation_as_terminal: bool = False
    # HL-Gauss classification critic (Farebrother et al. 2024,
    # arXiv:2403.03950): critic outputs hlgauss_num_bins logits over a fixed
    # support [hlgauss_vmin, hlgauss_vmax]; value targets are projected to a
    # truncated-Gaussian categorical (σ = hlgauss_sigma_ratio · bin_width) and
    # trained with cross-entropy; scalar V = expectation over bin centers.
    # Requires value_clip=false (no CE analog of PPO value clipping).
    critic_loss_mode: Literal["mse", "hlgauss"] = "mse"
    hlgauss_num_bins: int = 101
    hlgauss_vmin: Optional[float] = None
    hlgauss_vmax: Optional[float] = None
    hlgauss_sigma_ratio: float = 0.75


@dataclass
class GradientConfig:
    """Per-side multi-task gradient configuration.

    Each side (actor, critic) independently selects a combiner:

    - ``default``  — mean-average per-task slices.
    - ``pcgrad``   — PCGrad conflict-free projection.
    - ``cagrad``   — Conflict-Averse Gradient Descent QP.
    - ``fairgrad`` — FairGrad alpha-fairness KKT.

    When both sides are ``default`` the algorithm skips per-task backward
    loops.  When a non-default side projects on a shared trunk, that side
    owns the shared slot; if both are non-default, the actor wins.
    """
    actor_method:  Literal["default", "pcgrad", "cagrad", "fairgrad"] = "default"
    critic_method: Literal["default", "pcgrad", "cagrad", "fairgrad"] = "default"
    cagrad_alpha: float = 0.5
    # Per-side FairGrad alpha.  Actor 2.0 matches the official FairGrad
    # MetaWorld setting; critic 1.0 (proportional weighting) is gentler on
    # value heads.
    fairgrad_actor_alpha: float = 2.0
    fairgrad_critic_alpha: float = 1.0


@dataclass
class LoggingConfig:
    use_wandb: bool = True
    use_tensorboard: bool = False
    use_csv: bool = True
    train_interval: int = 2000
    test_interval: int = 1
    update_interval: int = 1
    info_interval: int = 1
    save_interval: int = 0
    save_best: bool = True
    log_dir: str = "logs"
    wandb_group: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    # Per-layer param/grad histograms are high-volume and rarely useful;
    # scalar grad norms already capture the signal.
    log_weights_grads_histograms: bool = False
    # Per-task gradient snapshot dump.  When ``dump_grads_dir`` is set, the
    # algorithm writes one ``.npz`` per gated minibatch (first repeat, first
    # minibatch of every ``dump_grads_every_collects`` collects) under that
    # directory, plus a single ``manifest.json``.  Each dump caches the raw
    # per-task flat gradients (actor view + critic-full view), per-task
    # PopArt (sigma, mu, W, b) for offline sigma^-2 inversion, and per-side
    # FairGrad diagnostics, so analysis scripts can rebuild Gram matrices,
    # eigenspectra, and offline FairGrad replays for arbitrary alpha without
    # rerunning training.  The literal value "auto" resolves to
    # ``<exp_dir>/grad_snapshots`` once the logger is built.
    # Storage: ~137 MB / snapshot at MT50 fp32; keep the cadence >= 5 in long
    # runs.  Off by default (``dump_grads_dir = None``) so baselines are
    # bit-identical and incur zero IO.
    dump_grads_dir: Optional[str] = None
    dump_grads_every_collects: int = 5


@dataclass
class Config:
    """Complete configuration for multi-task RL experiments."""

    project_name: str = "toppo-repro"  # W&B project; not the published liyp1220-uci/TOPPO
    exp_name: str = "mtrl_experiment"
    seed: int = 0
    device: str = "cuda"

    algo: Literal["ppo"] = "ppo"
    use_linear_baseline: bool = False

    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    gradient: GradientConfig = field(default_factory=GradientConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    config: Optional[str] = None


_NESTED_CONFIGS: dict[str, type] = {
    "env": EnvConfig,
    "network": NetworkConfig,
    "training": TrainingConfig,
    "ppo": PPOConfig,
    "gradient": GradientConfig,
    "logging": LoggingConfig,
}


def _matches_type(value, tp) -> bool:
    """Whether a YAML-loaded *value* fits the field annotation *tp*.

    Covers the annotation forms the config dataclasses use (str, int, float,
    bool, Literal, Optional, List).  A bool is not an int, an int is accepted
    where a float is expected, and any other annotation is let through.
    """
    origin = get_origin(tp)
    if origin is Union:
        return any(_matches_type(value, a) for a in get_args(tp))
    if origin is Literal:
        return any(value == a and type(value) is type(a) for a in get_args(tp))
    if origin is list:
        item = (get_args(tp) or (object,))[0]
        return isinstance(value, list) and all(_matches_type(v, item) for v in value)
    if tp is type(None):
        return value is None
    if tp is bool:
        return isinstance(value, bool)
    if tp is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if tp is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if tp is str:
        return isinstance(value, str)
    return True


def _check_fields(data: dict, cls: type, prefix: str = "") -> list:
    """One message per unknown key or wrongly typed value in *data*."""
    hints = get_type_hints(cls)
    errors = []
    for key, value in data.items():
        name = prefix + str(key)
        if key not in hints:
            errors.append(f"unknown key '{name}'")
        elif cls is Config and key in _NESTED_CONFIGS:
            if isinstance(value, dict):
                errors += _check_fields(value, _NESTED_CONFIGS[key], prefix=f"{key}.")
            elif value is not None:                 # an empty section means defaults
                errors.append(f"'{name}' must be a mapping, got {value!r}")
        elif not _matches_type(value, hints[key]):
            tp = hints[key]
            expected = tp.__name__ if isinstance(tp, type) else str(tp).replace("typing.", "")
            msg = f"'{name}' expects {expected}, got {type(value).__name__} {value!r}"
            if isinstance(value, str) and ("float" in expected or "int" in expected):
                try:
                    float(value)
                except ValueError:
                    pass
                else:
                    msg += (" (a number written as a string: drop any quotes, and write"
                            " an exponent with a dot and a signed power, e.g. 3.0e-4"
                            " or 1.0e+6, because YAML reads 3e-4 and 1.0e6 as"
                            " strings)")
            errors.append(msg)
    return errors


def _merge_yaml_into_config(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}
    # Reject rather than drop: a misspelled key would otherwise leave its field
    # at the default without a word, and a quoted number would reach the
    # trainer as a string.
    errors = (_check_fields(data, Config) if isinstance(data, dict)
              else [f"top level must be a mapping, got {type(data).__name__}"])
    if errors:
        raise ValueError(f"Invalid config file {yaml_path}:\n  " + "\n  ".join(errors))
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_config_from_dict(defaults: dict) -> Config:
    # Keys and value types were checked file by file in _merge_yaml_into_config.
    filtered = dict(defaults)
    for group_name, cls in _NESTED_CONFIGS.items():
        if group_name in filtered:
            default_data = asdict(cls())
            default_data.update(filtered[group_name] or {})
            filtered[group_name] = cls(**default_data)
    return Config(**filtered)


def load_config(yaml_path: Optional[str] = None, cli_args: Optional[list] = None) -> Config:
    """Load configuration from YAML then overlay with CLI args via Tyro.

    Precedence: CLI args > YAML file > defaults.  A YAML key that is not a
    config field, or a value of the wrong type, raises ValueError.
    """
    import tyro

    defaults: dict = {}
    if yaml_path is not None:
        defaults = _merge_yaml_into_config(yaml_path)

    args = cli_args if cli_args is not None else sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            yaml_defaults = _merge_yaml_into_config(args[i + 1])
            defaults = _deep_merge(defaults, yaml_defaults)
            break

    default_config = _build_config_from_dict(defaults) if defaults else Config()
    return tyro.cli(Config, default=default_config, args=args)


def save_config(config: Config, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
