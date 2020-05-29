import logging
import os
from pathlib import Path

from typing import Text, Dict, Any, List

from ruamel import yaml

from rasa.cli import utils as cli_utils
from rasa.constants import CONFIG_AUTOCONFIGURABLE_KEYS
from rasa.utils import io as io_utils

logger = logging.getLogger(__name__)


# In future iterations, this function and the function `create_config_for_keys` below
# will get additional input parameters to allow for smarter configuration. E.g.
# training_data: TrainingData
# domain: Domain
# stories: StoryGraph
def get_autoconfiguration(config_file) -> Dict[Text, Any]:
    """Determine configuration from a configuration file.

    Keys that are provided in the file are kept. Keys that are not provided are
    configured automatically.
    """
    if config_file and os.path.exists(config_file):
        config = io_utils.read_config_file(config_file)

        missing_keys = [k for k in CONFIG_AUTOCONFIGURABLE_KEYS if not config.get(k)]
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

    for key in keys:
        config[key] = default_config[key]
        # maybe add a debug output to print the result of the autoconfiguration?


def dump_config(config: Dict[Text, Any], config_file: Text) -> None:
    """Dump the automatically configured keys into the config file."""
    set_language = False

    try:
        content = io_utils.read_file(config_file, io_utils.DEFAULT_ENCODING)
    except ValueError:
        cli_utils.print_warning(
            f"Configuration file {config_file} does not exist. "
            f"Creating it now and filling it with the current "
            f"configuration."
        )
        content = {}
        set_language = True

    [content, yaml_parser] = io_utils.read_yaml(
        content, typ="rt", add_version=False, return_parser=True,
    )

    if set_language:
        content["language"] = config.get("language")

    autoconfigured = set()
    for key in CONFIG_AUTOCONFIGURABLE_KEYS:
        if not content.get(key):
            autoconfigured.add(key)
            content[key] = config.get(key)
            ct = yaml.tokens.CommentToken(
                f"# Configuration for {key} was provided by "
                f"the auto configuration.\n",
                yaml.error.CommentMark(0),
                None,
            )
            item = content.ca.items[key]
            if item[1]:
                item[1].append(ct)
            else:
                item[1] = [ct]

    if autoconfigured:
        cli_utils.print_info(
            f"Automatically configured {autoconfigured}. The configuration was written "
            f"into the config file at {config_file}"
        )

    yaml_parser.indent(mapping=2, sequence=4, offset=2)
    yaml_parser.dump(content, Path(config_file))
