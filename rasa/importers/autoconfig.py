import copy
import logging
import os
import re
import shutil
from pathlib import Path
import tempfile

from typing import Text, Dict, Any, List

from rasa.cli import utils as cli_utils
from rasa.constants import (
    CONFIG_AUTOCONFIGURABLE_KEYS,
    DOCS_URL_PIPELINE,
    DOCS_URL_POLICIES,
    CONFIG_KEYS,
)
from rasa.utils import io as io_utils

logger = logging.getLogger(__name__)


def get_configuration(config_file: Text) -> Dict[Text, Any]:
    """Determine configuration from a configuration file.

    Keys that are provided in the file are kept. Keys that are not provided are
    configured automatically.

    Args:
        config_file: The path to the configuration file.
    """
    if config_file and os.path.exists(config_file):
        config = io_utils.read_config_file(config_file)

        missing_keys = [k for k in CONFIG_AUTOCONFIGURABLE_KEYS if not config.get(k)]

        if missing_keys:
            config = _auto_configure(config, missing_keys)
            _dump_config(config, config_file)

    else:
        config = {}

    return config


def _auto_configure(config: Dict[Text, Any], keys: List[Text]) -> Dict[Text, Any]:
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


def _dump_config(config: Dict[Text, Any], config_file: Text) -> None:
    """Dump the automatically configured keys into the config file.

    The configuration provided in the file is kept as it is (preserving the order of
    keys and comments).
    For keys that were automatically configured, an explanatory comment is added and the
    automatically chosen configuration is added commented-out.
    If there are already blocks with comments from a previous auto configuration run,
    they are replaced with the new auto configuration.

    Args:
        config: The configuration including the automatically configured keys.
        config_file: The file into which the configuration should be dumped.
    """
    import pkg_resources

    try:
        content = io_utils.read_config_file(config_file)
    except ValueError:
        content = ""

    empty = False
    if not content:
        empty = True
        cli_utils.print_warning(
            f"Configuration file {config_file} does not exist or is empty or invalid. "
            f"Creating a new one now and filling it with the current configuration."
        )
        empty_config_file = pkg_resources.resource_filename(
            "rasa.cli.initial_project", "config.yml"
        )
        content = io_utils.read_config_file(empty_config_file)
        shutil.copy(empty_config_file, config_file)

    missing_keys = [k for k in CONFIG_KEYS if k not in content.keys()]
    autoconfigured = [k for k in CONFIG_AUTOCONFIGURABLE_KEYS if not content.get(k)]

    _add_missing_config_keys(missing_keys, config_file)

    lines_to_insert = _get_lines_to_insert(config, autoconfigured)

    try:
        with open(config_file, "r+", encoding=io_utils.DEFAULT_ENCODING) as f:
            lines = f.readlines()
            f.seek(0)

            remove_comments_until_next_uncommented_line = False
            for line in lines:
                insert_section = None

                if empty and config.get("language") and re.match("language:", line):
                    line = f"language: {config['language']}\n"

                if remove_comments_until_next_uncommented_line:
                    if re.match("#", line):  # old auto config to be removed
                        continue
                    remove_comments_until_next_uncommented_line = False

                for key in autoconfigured:
                    if re.match(f"{key}:( *)", line):  # start of next auto-section
                        line = line + _config_comment(key)
                        insert_section = key
                        remove_comments_until_next_uncommented_line = True
                f.write(line)

                if not insert_section:
                    continue

                # add the configuration (commented out)
                for line_to_insert in lines_to_insert[insert_section]:
                    f.write(line_to_insert)

    except FileNotFoundError:
        raise ValueError(f"File '{config_file}' does not exist.")

    if autoconfigured:
        autoconfigured_keys = cli_utils.transform_collection_to_sentence(autoconfigured)
        cli_utils.print_info(
            f"The configuration for {autoconfigured_keys} was chosen automatically. It "
            f"was written into the config file at `{config_file}`."
        )


def _add_missing_config_keys(missing_keys: List, config_file: Text) -> None:
    if missing_keys:
        with open(config_file, "a", encoding=io_utils.DEFAULT_ENCODING) as f:
            for key in missing_keys:
                f.write(f"{key}:\n")


def _get_lines_to_insert(
    config: Dict[Text, Any], autoconfigured: List
) -> Dict[Text, List[Text]]:
    import ruamel.yaml as yaml

    yaml_parser = yaml.YAML()
    yaml_parser.indent(mapping=2, sequence=4, offset=2)

    lines_to_insert = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for key in autoconfigured:
            file = Path(tmp_dir + f"/temp_{key}.yml")
            file.touch()
            yaml_parser.dump(config.get(key), file)

            try:
                with open(file) as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        lines[i] = "# " + line
                    lines_to_insert[key] = lines
            except FileNotFoundError:
                raise ValueError(f"File {file} does not exist.")
    return lines_to_insert


def _config_comment(key: Text) -> Text:
    if key == "pipeline":
        comment = (
            f"# # No configuration for the NLU pipeline was provided. The following "
            f"default pipeline was used to train your model.\n"
            f"# # To customise it, uncomment and adjust the pipeline. See "
            f"{DOCS_URL_PIPELINE} for more information.\n"
        )
    else:
        comment = (
            f"# # No configuration for policies was provided. The following default "
            f"policies were used to train your model.\n"
            f"# # To customise them, uncomment and adjust the policies. See "
            f"{DOCS_URL_POLICIES} for more information.\n"
        )
    return comment
