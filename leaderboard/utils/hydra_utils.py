from typing import List


def to_hydra_cli_args(config: dict, prefix: str = "") -> List[str]:
    """Convert the dict representation of a configuration to hydra command line args.

    Args:
        config: dict representation of an experiment configuration
        prefix: prefix to be used for all keys of the given config
    """
    args = []
    for key, value in config.items():
        if hasattr(value, "items"):  # omegaconf.dictconfig.DictConfig vs. dict
            args += to_hydra_cli_args(value, prefix=f"{prefix}{key}.")
        else:
            args += [f"{prefix}{key}='{value}'"]
    return args
