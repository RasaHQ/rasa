import copy
import logging
import os
import re
import shutil
from pathlib import Path
import tempfile

from typing import Text, Dict, Any, List, Set

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
        config = _create_config_for_keys(config, missing_keys)

        dump_config(config, config_file)
    else:
        config = {}

    return config


def _create_config_for_keys(
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
        shutil.copy(empty_config_file, config_file)

    content, yaml_parser = io_utils.read_yaml_including_parser(
        content, typ="rt", add_version=False
    )
    yaml_parser.indent(mapping=2, sequence=4, offset=2)

    with tempfile.TemporaryDirectory() as tmp_dir:

        autoconfigured = set()
        for key in CONFIG_AUTOCONFIGURABLE_KEYS:
            if content.get(key):
                continue
            autoconfigured.add(key)
            file = Path(tmp_dir + f"/temp_{key}.yml")
            file.touch()
            yaml_parser.dump(config.get(key), file)

        try:
            with open(config_file, "r+", encoding=io_utils.DEFAULT_ENCODING) as f:
                lines = f.readlines()
                f.seek(0)

                removing_old_config = False
                for line in lines:
                    insert_section = None

                    if empty and re.match("language:", line):
                        line = f"language: {config['language']}\n"

                    if removing_old_config:
                        if re.match(f"#", line):  # old auto config to be removed
                            continue
                        else:
                            removing_old_config = False

                    for key in autoconfigured:
                        if re.match(f"( *){key}:( *)", line):  # start of next section
                            comment = (
                                f"# Configuration for {key} was provided by the auto "
                                f"configuration.\n"
                                f"# To configure it manually, uncomment this section's "
                                f"content.\n"
                                f"#\n"
                            )
                            line = line + comment
                            insert_section = key
                            removing_old_config = True
                    f.write(line)

                    if not insert_section:
                        continue

                    section_file = tmp_dir + f"/temp_{insert_section}.yml"
                    print(Path(section_file).is_file())
                    try:
                        with open(section_file) as sub_f:
                            section_lines = sub_f.readlines()
                            for section_line in section_lines:
                                f.write("#" + section_line)
                    except FileNotFoundError:
                        raise ValueError(f"File {section_file} does not exist.")

        except FileNotFoundError:
            raise ValueError(f"File '{config_file}' does not exist.")

    if autoconfigured:
        cli_utils.print_info(
            f"Configuration of {autoconfigured} was chosen automatically. It was "
            f"written into the config file at {config_file}."
        )
