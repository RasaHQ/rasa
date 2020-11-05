import copy
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Text, Dict, Any, List, Set, Optional

import rasa.shared.constants
from rasa.shared.exceptions import FileNotFoundException
import rasa.shared.utils.cli
import rasa.shared.utils.common
import rasa.shared.utils.io

logger = logging.getLogger(__name__)

COMMENTS_FOR_KEYS = {
    "pipeline": (
        f"# # No configuration for the NLU pipeline was provided. The following "
        f"default pipeline was used to train your model.\n"
        f"# # If you'd like to customize it, uncomment and adjust the pipeline.\n"
        f"# # See {rasa.shared.constants.DOCS_URL_PIPELINE} for more information.\n"
    ),
    "policies": (
        f"# # No configuration for policies was provided. The following default "
        f"policies were used to train your model.\n"
        f"# # If you'd like to customize them, uncomment and adjust the policies.\n"
        f"# # See {rasa.shared.constants.DOCS_URL_POLICIES} for more information.\n"
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

    config = rasa.shared.utils.io.read_config_file(config_file_path)

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
        all_keys = rasa.shared.constants.CONFIG_AUTOCONFIGURABLE_KEYS_NLU
    elif training_type == TrainingType.CORE:
        all_keys = rasa.shared.constants.CONFIG_AUTOCONFIGURABLE_KEYS_CORE
    else:
        all_keys = rasa.shared.constants.CONFIG_AUTOCONFIGURABLE_KEYS

    return {k for k in all_keys if not config.get(k)}


def _get_missing_config_keys(
    config: Dict[Text, Any], training_type: Optional[TrainingType] = TrainingType.BOTH
) -> Set[Text]:
    if training_type == TrainingType.NLU:
        all_keys = rasa.shared.constants.CONFIG_KEYS_NLU
    elif training_type == TrainingType.CORE:
        all_keys = rasa.shared.constants.CONFIG_KEYS_CORE
    else:
        all_keys = rasa.shared.constants.CONFIG_KEYS

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
            f"{rasa.shared.utils.common.transform_collection_to_sentence(keys_to_configure)}. "
            f"Values will be provided from the default configuration."
        )

    filename = "default_config.yml"

    default_config_file = pkg_resources.resource_filename(__name__, filename)
    default_config = rasa.shared.utils.io.read_config_file(default_config_file)

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
        config_file_path, missing_keys, auto_configured_keys, training_type
    )
    if not config_as_expected:
        rasa.shared.utils.cli.print_error(
            f"The configuration file at '{config_file_path}' has been removed or "
            f"modified while the automatic configuration was running. The current "
            f"configuration will therefore not be dumped to the file. If you want to "
            f"your model to use the configuration provided in '{config_file_path}', "
            f"you need to re-run training."
        )
        return

    _add_missing_config_keys_to_file(config_file_path, missing_keys)

    autoconfig_lines = _get_commented_out_autoconfig_lines(config, auto_configured_keys)

    current_config_content = rasa.shared.utils.io.read_file(config_file_path)
    current_config_lines = current_config_content.splitlines(keepends=True)

    updated_lines = _get_lines_including_autoconfig(
        current_config_lines, autoconfig_lines
    )

    rasa.shared.utils.io.write_text_file("".join(updated_lines), config_file_path)

    auto_configured_keys = rasa.shared.utils.common.transform_collection_to_sentence(
        auto_configured_keys
    )
    rasa.shared.utils.cli.print_info(
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
        content = rasa.shared.utils.io.read_config_file(config_file_path)
    except FileNotFoundException:
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
    with open(
        config_file_path, "a", encoding=rasa.shared.utils.io.DEFAULT_ENCODING
    ) as f:
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
