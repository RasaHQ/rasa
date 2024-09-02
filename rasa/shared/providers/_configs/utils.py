import structlog
from rasa.shared.utils.io import raise_deprecation_warning

structlogger = structlog.get_logger()


def resolve_aliases(config: dict, deprecated_alias_mapping: dict) -> dict:
    """
    Resolve aliases in the configuration to standard keys.

    Args:
        config: Dictionary containing the configuration.
        deprecated_alias_mapping: Dictionary mapping aliases to
            their standard keys.

    Returns:
        New dictionary containing the processed configuration.
    """
    config = config.copy()

    for alias, standard_key in deprecated_alias_mapping.items():
        # We check for the alias instead of the standard key because our goal is to
        # update the standard key when the alias is found. Since the standard key is
        # always included in the default component configurations, we overwrite it
        # with the alias value if the alias exists.
        if alias in config:
            config[standard_key] = config.pop(alias)

    return config


def raise_deprecation_warnings(config: dict, deprecated_alias_mapping: dict) -> None:
    """
    Raises warnings for deprecated keys in the configuration.

    Args:
        config: Dictionary containing the configuration.
        deprecated_alias_mapping: Dictionary mapping deprecated keys to
            their standard keys.

    Raises:
        DeprecationWarning: If any deprecated key is found in the config.
    """
    for alias, standard_key in deprecated_alias_mapping.items():
        if alias in config:
            raise_deprecation_warning(
                message=(
                    f"'{alias}' is deprecated and will be removed in "
                    f"4.0.0. Use '{standard_key}' instead."
                )
            )


def validate_required_keys(config: dict, required_keys: list) -> None:
    """
    Validates that the passed config contains all the required keys.

    Args:
        config: Dictionary containing the configuration.
        required_keys: List of keys that must be present in the config.

    Raises:
        ValueError: If any required key is missing.
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        message = f"Missing required keys '{missing_keys}' for configuration."
        structlogger.error(
            "validate_required_keys",
            message=message,
            missing_keys=missing_keys,
            config=config,
        )
        raise ValueError(message)


def validate_forbidden_keys(config: dict, forbidden_keys: list) -> None:
    """
    Validates that the passed config doesn't contain any forbidden keys.

    Args:
        config: Dictionary containing the configuration.
        forbidden_keys: List of keys that are forbidden in the config.

    Raises:
        ValueError: If any forbidden key is present.
    """
    forbidden_keys_in_config = set(config.keys()).intersection(set(forbidden_keys))

    if forbidden_keys_in_config:
        message = (
            f"Forbidden keys '{forbidden_keys_in_config}' present "
            f"in the configuration."
        )
        structlogger.error(
            "validate_forbidden_keys",
            message=message,
            forbidden_keys=forbidden_keys_in_config,
            config=config,
        )
        raise ValueError(message)
