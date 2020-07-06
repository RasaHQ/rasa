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
from rasa.utils import io as io_utils, common as common_utils

logger = logging.getLogger(__name__)

COMMENTS_FOR_KEYS = {
    "pipeline": (
        f"# # No configuration for the NLU pipeline was provided. The following "
        f"default pipeline was used to train your model.\n"
        f"# # To customise it, uncomment and adjust the pipeline. See "
        f"{DOCS_URL_PIPELINE} for more information.\n"
    ),
    "policies": (
        f"# # No configuration for policies was provided. The following default "
        f"policies were used to train your model.\n"
        f"# # To customise them, uncomment and adjust the policies. See "
        f"{DOCS_URL_POLICIES} for more information.\n"
    ),
}


def _get_unspecified_autoconfigurable_keys(config: Dict[Text, Any]) -> List[Text]:
    return [k for k in CONFIG_AUTOCONFIGURABLE_KEYS if not config.get(k)]


def _get_missing_config_keys(config: Dict[Text, Any]) -> List[Text]:
    return [k for k in CONFIG_KEYS if k not in config.keys()]


def get_configuration(config_file_path: Text) -> Dict[Text, Any]:
    """Determine configuration from a configuration file.

    Keys that are provided in the file are kept. Keys that are not provided are
    configured automatically.

    Args:
        config_file_path: The path to the configuration file.
    """
    if config_file_path and os.path.exists(config_file_path):
        config = io_utils.read_config_file(config_file_path)

        missing_keys = _get_unspecified_autoconfigurable_keys(config)

        if missing_keys:
            config = _auto_configure(config, missing_keys)
            _dump_config(config, config_file_path)

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


def _dump_config(config: Dict[Text, Any], config_file_path: Text) -> None:
    """Dump the automatically configured keys into the config file.

    The configuration provided in the file is kept as it is (preserving the order of
    keys and comments).
    For keys that were automatically configured, an explanatory comment is added and the
    automatically chosen configuration is added commented-out.
    If there are already blocks with comments from a previous auto configuration run,
    they are replaced with the new auto configuration.

    Args:
        config: The configuration including the automatically configured keys.
        config_file_path: The file into which the configuration should be dumped.
    """
    try:
        content = io_utils.read_config_file(config_file_path)
    except ValueError:
        content = ""

    language_to_overwrite = None
    if not content:
        content = _create_and_read_config_file(config_file_path)
        # if the config file was empty or not present, the default language will be
        # overwritten with the language in the current config
        language_to_overwrite = config.get("language")

    missing_keys = _get_missing_config_keys(content)
    _add_missing_config_keys_to_file(missing_keys, config_file_path)

    autoconfigured = _get_unspecified_autoconfigurable_keys(content)
    autoconfig_lines = _get_commented_out_autoconfig_lines(config, autoconfigured)

    try:
        with open(config_file_path, "r+", encoding=io_utils.DEFAULT_ENCODING) as f:
            lines = f.readlines()
            updated_lines = _get_lines_including_autoconfig(
                lines, autoconfig_lines, language_to_overwrite
            )
            f.seek(0)
            for line in updated_lines:
                f.write(line)
    except FileNotFoundError:
        raise ValueError(f"File '{config_file_path}' does not exist.")

    if autoconfigured:
        autoconfigured_keys = common_utils.transform_collection_to_sentence(
            autoconfigured
        )
        cli_utils.print_info(
            f"The configuration for {autoconfigured_keys} was chosen automatically. It "
            f"was written into the config file at `{config_file_path}`."
        )


def _create_and_read_config_file(config_file_path: Text) -> Dict[Text, Any]:
    import pkg_resources

    cli_utils.print_warning(
        f"Configuration file {config_file_path} does not exist or is empty or invalid. "
        f"Creating a new one now and filling it with the current configuration."
    )

    empty_config_file = pkg_resources.resource_filename(
        "rasa.cli.initial_project", "config.yml"
    )
    shutil.copy(empty_config_file, config_file_path)

    return io_utils.read_config_file(config_file_path)


def _add_missing_config_keys_to_file(
    missing_keys: List, config_file_path: Text
) -> None:
    if missing_keys:
        with open(config_file_path, "a", encoding=io_utils.DEFAULT_ENCODING) as f:
            for key in missing_keys:
                f.write(f"{key}:\n")


def _get_lines_including_autoconfig(
    lines: List[Text],
    autoconfig_lines: Dict[Text, List[Text]],
    language_to_overwrite: Text = None,
) -> List[Text]:
    autoconfigured = autoconfig_lines.keys()

    lines_with_autoconfig = []
    remove_comments_until_next_uncommented_line = False
    for line in lines:
        insert_section = None

        # overwrite language if necessary
        if language_to_overwrite and re.match("language:", line):
            line = f"language: {language_to_overwrite}\n"

        # remove old auto configuration
        if remove_comments_until_next_uncommented_line:
            if re.match("#", line):
                continue
            remove_comments_until_next_uncommented_line = False

        # add an explanatory comment to auto configured sections
        for key in autoconfigured:
            if re.match(f"{key}:( *)", line):  # start of next auto-section
                line = line + COMMENTS_FOR_KEYS[key]
                insert_section = key
                remove_comments_until_next_uncommented_line = True

        lines_with_autoconfig.append(line)

        if not insert_section:
            continue

        # add the auto configuration (commented out)
        lines_with_autoconfig += autoconfig_lines[insert_section]

    return lines_with_autoconfig


def _get_commented_out_autoconfig_lines(
    config: Dict[Text, Any], autoconfigured: List[Text]
) -> Dict[Text, List[Text]]:
    import ruamel.yaml as yaml

    yaml_parser = yaml.YAML()
    yaml_parser.indent(mapping=2, sequence=4, offset=2)

    autoconfig_lines = {}

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
                    autoconfig_lines[key] = lines
            except FileNotFoundError:
                raise ValueError(f"File {file} does not exist.")
    return autoconfig_lines
