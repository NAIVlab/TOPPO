import wandb
import argparse
from importlib import import_module


def run_experiment(config, seeds, main_module_name):
    main_module = import_module(main_module_name)
    for seed in seeds:
        config.seed = seed
        main_module.main(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MTRL experiments with different seeds and main modules."
    )
    parser.add_argument(
        "--main_module_name",
        type=str,
        required=True,
        help="Name of the main script to use (e.g., 'main_script', 'main_script_v2').",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="List of seeds for experiments.",
    )
    parser.add_argument(
        "--config_args",
        type=str,
        nargs="*",
        help="List of key=value pairs to update the config parameters.",
    )
    args = parser.parse_args()

    # Extract arguments
    main_module_name = args.main_module_name
    seeds = args.seeds
    config_args = args.config_args

    # Define initial config
    main_module = import_module(main_module_name)
    config = main_module.Config()

    if config_args:
        for arg in config_args:
            key, value = arg.split("=")
            try:
                value = eval(value)
            except:
                pass

            setattr(config, key, value)
    config.__post_init__()

    run_experiment(config, seeds, main_module_name)

    wandb.finish()
