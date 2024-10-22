import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Text

from rasa.shared.exceptions import RasaException, YamlException
from rasa.shared.utils.yaml import (
    read_config_file,
    read_yaml_file,
    validate_yaml_content_using_schema,
)

logger = logging.getLogger(__name__)
SCHEMA_FILE = Path(__file__).parent / "schemas" / "config.yml"


def read_endpoint_config(
    filename: Optional[Text], endpoint_type: Text
) -> Optional[Dict[Text, Any]]:
    """Reads a configuration file from disk and extract one config."""
    if not filename:
        return None

    try:
        content = read_config_file(filename, reader_type=["safe", "rt"])

        if content.get(endpoint_type) is None:
            return None

        return content
    except FileNotFoundError:
        logger.error(
            "Failed to read configuration from {}. No such file.".format(
                os.path.abspath(filename)
            ),
        )
        return None


def extract_anonymization_traits(
    anonymization_config: Optional[Dict[Text, Any]], endpoint_type: Text
) -> Dict[Text, Any]:
    """Extract anonymization traits from the anonymization config.

    Args:
        anonymization_config: The anonymization config.
        endpoint_type: The endpoint type.

    Returns:
        A dictionary containing the anonymization traits.
    """
    if not anonymization_config:
        return {"enabled": False}

    anonymization_value = anonymization_config.get(endpoint_type)
    if not anonymization_value:
        logger.debug("Anonymization config found but nothing is defined.")
        return {"enabled": False}

    try:
        validate_anonymization_yaml(anonymization_config)
    except RasaException as exception:
        logger.debug(exception)
        return {"enabled": False}

    rule_lists = anonymization_value.get("rule_lists", [])

    traits = {
        "enabled": True,
        "metadata": anonymization_value.get("metadata", {}),
        "number_of_rule_lists": len(rule_lists),
        "number_of_rules": 0,
        "substitutions": {
            "mask": 0,
            "faker": 0,
            "text": 0,
            "not_defined": 0,
        },
        "entities": set(),
    }

    for rule_list in rule_lists:
        traits["number_of_rules"] += len(rule_list.get("rules", []))
        for rule in rule_list.get("rules", []):
            traits["substitutions"][rule.get("substitution", "not_defined")] += 1
            traits["entities"].add(rule.get("entity"))

    entities = list(traits["entities"])
    entities.sort()
    traits["entities"] = entities

    return traits


def validate_anonymization_yaml(yaml_content: Dict[Text, Any]) -> None:
    """Checks if the yaml_content adheres to the anonymization rule schema.

    If the yaml_content is not in the right format, an exception will be raised.
    """
    schema = read_yaml_file(SCHEMA_FILE, reader_type=("safe", "rt"))
    try:
        validate_yaml_content_using_schema(yaml_content, schema)
    except YamlException as exception:
        raise RasaException(
            f"Invalid configuration for `anonymization_rules` : {exception}"
        ) from exception

    rule_lists = yaml_content.get("anonymization", {}).get("rule_lists", [])

    rule_set = set()
    for rule_list in rule_lists:
        rule_id = rule_list.get("id")
        if rule_id in rule_set:
            raise RasaException(
                f"Invalid configuration for `anonymization_rules` :"
                f"Duplicate rule id: '{rule_id}' encountered in rule_lists"
            )
        rule_set.add(rule_id)
