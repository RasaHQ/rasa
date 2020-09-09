import copy
import logging
import os
import sys
from enum import Enum
from typing import Text, Dict, Any, List, Set, Optional

import rasa.constants as constants
import rasa.utils.common as common_utils
import rasa.cli.utils as cli_utils
import rasa.utils.io as io_utils

logger = logging.getLogger(__name__)

COMMENTS_FOR_KEYS = {
    "pipeline": (
        f"# # No configuration for the NLU pipeline was provided. The following "
        f"default pipeline was used to train your model.\n"
        f"# # If you'd like to customize it, uncomment and adjust the pipeline.\n"
        f"# # See {constants.DOCS_URL_PIPELINE} for more information.\n"
    ),
    "policies": (
        f"# # No configuration for policies was provided. The following default "
        f"policies were used to train your model.\n"
        f"# # If you'd like to customize them, uncomment and adjust the policies.\n"
        f"# # See {constants.DOCS_URL_POLICIES} for more information.\n"
    ),
}


class TrainingType(Enum):
    NLU = 1
    CORE = 2
    BOTH = 3


def get_configuration(
    config_file_path: Text, training_type: Optional[TrainingType] = TrainingType.BOTH
) -> Dict[Text, Any]:
    """Determine configuration from a configuration file.

    Keys that are provided and have a value in the file are kept. Keys that are not
    provided are configured automatically.

    Args:
        config_file_path: The path to the configuration file.
        training_type: NLU, CORE or BOTH depending on what is trained.
    """
    if not config_file_path or not os.path.exists(config_file_path):
        logger.debug("No configuration file was provided to the TrainingDataImporter.")
        return {}

    config = io_utils.read_config_file(config_file_path)

    missing_keys = _get_missing_config_keys(config, training_type)
    keys_to_configure = _get_unspecified_autoconfigurable_keys(config, training_type)

    if keys_to_configure:
        config = _auto_configure(config, keys_to_configure)
        _dump_config(
            config, config_file_path, missing_keys, keys_to_configure, training_type
        )

    return config


def _get_unspecified_autoconfigurable_keys(
    config: Dict[Text, Any], training_type: Optional[TrainingType] = TrainingType.BOTH
) -> Set[Text]:
    if training_type == TrainingType.NLU:
        all_keys = constants.CONFIG_AUTOCONFIGURABLE_KEYS_NLU
    elif training_type == TrainingType.CORE:
        all_keys = constants.CONFIG_AUTOCONFIGURABLE_KEYS_CORE
    else:
        all_keys = constants.CONFIG_AUTOCONFIGURABLE_KEYS

    return {k for k in all_keys if not config.get(k)}


def _get_missing_config_keys(
    config: Dict[Text, Any], training_type: Optional[TrainingType] = TrainingType.BOTH
) -> Set[Text]:
    if training_type == TrainingType.NLU:
        all_keys = constants.CONFIG_KEYS_NLU
    elif training_type == TrainingType.CORE:
        all_keys = constants.CONFIG_KEYS_CORE
    else:
        all_keys = constants.CONFIG_KEYS

    return {k for k in all_keys if k not in config.keys()}


def _auto_configure(
    config: Dict[Text, Any], keys_to_configure: Set[Text]
) -> Dict[Text, Any]:
    """Complete a config by adding automatic configuration for the specified keys.

    Args:
        config: The provided configuration.
        keys_to_configure: Keys to be configured automatically (e.g. `policies`).

    Returns:
        The resulting configuration including both the provided and the automatically
        configured keys.
    """
    import pkg_resources

    if keys_to_configure:
        logger.debug(
            f"The provided configuration does not contain the key(s) "
            f"{common_utils.transform_collection_to_sentence(keys_to_configure)}. "
            f"Values will be provided from the default configuration."
        )

    if sys.platform == "win32":
        filename = "default_config_other_language.yml"
    elif config.get("language") == "en":
        filename = "default_config_en.yml"
    else:
        filename = "default_config_other_language.yml"

    default_config_file = pkg_resources.resource_filename(__name__, filename)
    default_config = io_utils.read_config_file(default_config_file)

    config = copy.deepcopy(config)
    for key in keys_to_configure:
        config[key] = default_config[key]

    return config


def _dump_config(
    config: Dict[Text, Any],
    config_file_path: Text,
    missing_keys: Set[Text],
    auto_configured_keys: Set[Text],
    training_type: Optional[TrainingType] = TrainingType.BOTH,
) -> None:
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
        missing_keys: Keys that need to be added to the config file.
        auto_configured_keys: Keys for which a commented out auto configuration section
                              needs to be added to the config file.
        training_type: NLU, CORE or BOTH depending on which is trained.
    """

    config_as_expected = _is_config_file_as_expected(
        config_file_path, missing_keys, auto_configured_keys, training_type,
    )
    if not config_as_expected:
        cli_utils.print_error(
            f"The configuration file at '{config_file_path}' has been removed or "
            f"modified while the automatic configuration was running. The current "
            f"configuration will therefore not be dumped to the file. If you want to "
            f"your model to use the configuration provided in '{config_file_path}', "
            f"you need to re-run training."
        )
        return

    _add_missing_config_keys_to_file(config_file_path, missing_keys)

    autoconfig_lines = _get_commented_out_autoconfig_lines(config, auto_configured_keys)

    with open(config_file_path, "r+", encoding=io_utils.DEFAULT_ENCODING) as f:
        lines = f.readlines()
        updated_lines = _get_lines_including_autoconfig(lines, autoconfig_lines)
        f.seek(0)
        for line in updated_lines:
            f.write(line)

    auto_configured_keys = common_utils.transform_collection_to_sentence(
        auto_configured_keys
    )
    cli_utils.print_info(
        f"The configuration for {auto_configured_keys} was chosen automatically. It "
        f"was written into the config file at '{config_file_path}'."
    )


def _is_config_file_as_expected(
    config_file_path: Text,
    missing_keys: Set[Text],
    auto_configured_keys: Set[Text],
    training_type: Optional[TrainingType] = TrainingType.BOTH,
) -> bool:
    try:
        content = io_utils.read_config_file(config_file_path)
    except ValueError:
        content = ""

    return (
        bool(content)
        and missing_keys == _get_missing_config_keys(content, training_type)
        and auto_configured_keys
        == _get_unspecified_autoconfigurable_keys(content, training_type)
    )


def _add_missing_config_keys_to_file(
    config_file_path: Text, missing_keys: Set[Text]
) -> None:
    if not missing_keys:
        return
    with open(config_file_path, "a", encoding=io_utils.DEFAULT_ENCODING) as f:
        for key in missing_keys:
            f.write(f"{key}:\n")


def _get_lines_including_autoconfig(
    lines: List[Text], autoconfig_lines: Dict[Text, List[Text]]
) -> List[Text]:
    auto_configured_keys = autoconfig_lines.keys()

    lines_with_autoconfig = []
    remove_comments_until_next_uncommented_line = False
    for line in lines:
        insert_section = None

        # remove old auto configuration
        if remove_comments_until_next_uncommented_line:
            if line.startswith("#"):
                continue
            remove_comments_until_next_uncommented_line = False

        # add an explanatory comment to auto configured sections
        for key in auto_configured_keys:
            if line.startswith(f"{key}:"):  # start of next auto-section
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
    config: Dict[Text, Any], auto_configured_keys: Set[Text]
) -> Dict[Text, List[Text]]:
    import ruamel.yaml as yaml
    import ruamel.yaml.compat

    yaml_parser = yaml.YAML()
    yaml_parser.indent(mapping=2, sequence=4, offset=2)

    autoconfig_lines = {}

    for key in auto_configured_keys:
        stream = yaml.compat.StringIO()
        yaml_parser.dump(config.get(key), stream)
        dump = stream.getvalue()

        lines = dump.split("\n")
        if not lines[-1]:
            lines = lines[:-1]  # yaml dump adds an empty line at the end
        lines = [f"# {line}\n" for line in lines]

        autoconfig_lines[key] = lines

    return autoconfig_lines
