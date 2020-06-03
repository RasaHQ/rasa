import copy
import logging
import os
from pathlib import Path

from typing import Text, Dict, Any, List

from rasa.cli import utils as cli_utils
from rasa.constants import CONFIG_AUTOCONFIGURABLE_KEYS
from rasa.utils import io as io_utils

logger = logging.getLogger(__name__)


# In future iterations, this function and the function `create_config_for_keys` below
# will get additional input parameters to allow for smarter configuration. E.g.
# training_data: TrainingData
# domain: Domain
# stories: StoryGraph
def get_auto_configuration(config_file) -> Dict[Text, Any]:
    """Determine configuration from a configuration file.

    Keys that are provided in the file are kept. Keys that are not provided are
    configured automatically.
    """
    if config_file and os.path.exists(config_file):
        config = io_utils.read_config_file(config_file)

        missing_keys = [k for k in CONFIG_AUTOCONFIGURABLE_KEYS if not config.get(k)]
        config = create_config_for_keys(config, missing_keys)

        dump_config(config, config_file)
    else:
        config = {}

    return config


def create_config_for_keys(
    config: Dict[Text, Any], keys: List[Text]
) -> Dict[Text, Any]:
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

    config = copy.deepcopy(config)
    for key in keys:
        config[key] = default_config[key]
        # maybe add a debug output to print the result of the autoconfiguration?

    return config


def dump_config(config: Dict[Text, Any], config_file: Text) -> None:
    """Dump the automatically configured keys into the config file."""
    import pkg_resources

    try:
        content = io_utils.read_file(config_file, io_utils.DEFAULT_ENCODING)
    except ValueError:
        content = ""

    empty = False
    if not content:
        empty = True
        cli_utils.print_warning(
            f"Configuration file {config_file} does not exist or is empty. "
            f"Creating it now and filling it with the current configuration."
        )
        empty_config_file = pkg_resources.resource_filename(
            "rasa.cli.initial_project", "config.yml"
        )
        content = io_utils.read_file(empty_config_file, io_utils.DEFAULT_ENCODING)

    content, yaml_parser = io_utils.read_yaml_including_parser(
        content, typ="rt", add_version=False,
    )

    if empty:
        content["language"] = config.get("language")

    autoconfigured = set()
    for key in CONFIG_AUTOCONFIGURABLE_KEYS:
        if content.get(key):
            continue

        autoconfigured.add(key)
        content[key] = config.get(key)

        # needed to "fix" the comment structure for content loaded from an empty config
        item = content.ca.items.get(key)
        if item and item[1] is None:
            content.ca.items[key][1] = []

        comment = (
            f"Configuration for {key} was provided by the auto configuration.\n"
            f"To configure this manually, uncomment the section's content."
        )
        content.yaml_set_comment_before_after_key(key, before=comment)

    yaml_parser.indent(mapping=2, sequence=4, offset=2)
    yaml_parser.dump(content, Path(config_file))

    io_utils.comment_out_section(config_file, list(autoconfigured))

    if autoconfigured:
        cli_utils.print_info(
            f"Automatically configured {autoconfigured}. The configuration was written "
            f"into the config file at {config_file}"
        )
