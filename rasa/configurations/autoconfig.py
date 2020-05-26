import logging

from typing import Text, Dict, Any, List

from rasa.constants import CONFIG_AUTOCONFIGURABLE_KEYS
from rasa.utils import io as io_utils

logger = logging.getLogger(__name__)


# In future iterations, this function and the function `create_config_for_keys` below
# will get additional input parameters to allow for smarter configuration. E.g.
# training_data: TrainingData
# domain: Domain
# stories: StoryGraph
def get_autoconfiguration(config):
    autoconfig_keys = CONFIG_AUTOCONFIGURABLE_KEYS
    missing_keys = [k for k in autoconfig_keys if k not in config]

    create_config_for_keys(config, missing_keys)

    return config


def create_config_for_keys(config: Dict[Text, Any], keys: List[Text]) -> None:
    import pkg_resources

    if keys:
        logger.debug(
            f"The provided configuration does not contain the key(s) {keys}. "
            f"Running automatic configuration for them now."
        )

    default_config_path = pkg_resources.resource_filename(
        __name__, "default_config.yml"
    )

    default_config = io_utils.read_config_file(default_config_path)

    if not config.get("autoconfigured"):
        config["autoconfigured"] = set()

    for key in keys:
        config[key] = default_config[key]
        config["autoconfigured"].add(key)
        # maybe add a debug output to print the result of the autoconfiguration?


def dump_config(config: Dict[Text, Any], config_path: Text) -> None:
    # only write section for a key if that key is in config.autoconfigured
    io_utils.write_yaml_file(config, config_path)
