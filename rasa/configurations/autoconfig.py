import logging
import os

from typing import Text, Dict, Any, List

from rasa.cli import utils as cli_utils
from rasa.constants import CONFIG_AUTOCONFIGURABLE_KEYS, AUTOCONFIGURATION_KEY
from rasa.utils import io as io_utils

logger = logging.getLogger(__name__)


# In future iterations, this function and the function `create_config_for_keys` below
# will get additional input parameters to allow for smarter configuration. E.g.
# training_data: TrainingData
# domain: Domain
# stories: StoryGraph
def get_autoconfiguration(config_file):
    """Determine configuration from a configuration file.

    Keys that are provided in the file are kept. Keys that are not provided are
    configured automatically.
    """
    if config_file and os.path.exists(config_file):
        config = io_utils.read_config_file(config_file)

        missing_keys = [k for k in CONFIG_AUTOCONFIGURABLE_KEYS if k not in config]
        create_config_for_keys(config, missing_keys)

        dump_config(config, config_file)
    else:
        config = {}

    return config


def create_config_for_keys(config: Dict[Text, Any], keys: List[Text]) -> None:
    """Complete a config by adding automatic configuration for the specified keys.

    Args:
        config: The provided configuration.
        keys: Keys to be configured automatically (e.g. `policies`, `pipeline`).
    """
    import pkg_resources

    if keys:
        logger.debug(
            f"The provided configuration does not contain the key(s) {keys}. "
            f"Running automatic configuration for them now."
        )

    default_config_file = pkg_resources.resource_filename(
        __name__, "default_config.yml"
    )

    default_config = io_utils.read_config_file(default_config_file)

    if not config.get(AUTOCONFIGURATION_KEY):
        config[AUTOCONFIGURATION_KEY] = set()

    for key in keys:
        config[key] = default_config[key]
        config[AUTOCONFIGURATION_KEY].add(key)
        # maybe add a debug output to print the result of the autoconfiguration?


def dump_config(config: Dict[Text, Any], config_file: Text) -> None:
    """Dump the automatically configured keys into the config file."""
    # only write sections for keys that were autoconfigured
    new_sections = {}
    for key in config.get(AUTOCONFIGURATION_KEY):
        new_sections[key] = config.get(key)

    io_utils.change_sections_in_yaml_file(new_sections, config_file)

    cli_utils.print_info(
        f"Automatically configured {config.get(AUTOCONFIGURATION_KEY)}. The key(s) "
        f"was/were added to the config file at {config_file}"
    )
