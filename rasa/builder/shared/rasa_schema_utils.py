"""Shared domain schema utilities for CALM-style bots.

Provides a domain schema cleaned of NLU-specific parts (intents, entities, etc.)
and with responses schema merged in. Used by project generation and MCP schema
tools.
"""

import copy
from typing import Any, Dict, Optional, cast

import importlib_resources

from rasa.e2e_test.constants import SCHEMA_FILE_PATH as E2E_TESTS_SCHEMA_FILE
from rasa.shared.constants import (
    DOMAIN_SCHEMA_FILE,
    PACKAGE_NAME,
    RESPONSES_SCHEMA_FILE,
)
from rasa.shared.core.flows.yaml_flows_io import FLOWS_SCHEMA_FILE
from rasa.shared.utils.io import read_json_file
from rasa.shared.utils.yaml import read_schema_file

# Non-CALM properties in domain schema keys (intents, entities, forms, config,
# session_config).
NON_CALM_DOMAIN_KEYS = [
    "intents",
    "entities",
    "forms",
    "config",
    "session_config",
]
# Non-CALM properties in slot mapping entries (from_intent, from_entity, form-based).
# CALM keeps: type (from_llm/controlled), run_action_every_turn, coexistence_system,
# conditions.active_flow; and the whole validation section.
NON_CALM_SLOT_MAPPING_KEYS = [
    "intent",
    "not_intent",
    "entity",
    "role",
    "group",
    "value",
    "action",
]
NON_CALM_CONDITION_KEYS = ["active_loop", "requested_slot"]


def get_flows_schema(calm_only: bool = True) -> dict[str, Any]:
    """Load the flows YAML schema, optionally excluding the NLU trigger.

    Args:
        calm_only: If True (default), the schema excludes the NLU-related properties
            from flow definitions (CALM-style, intent-agnostic).
    """
    flows_schema_file_path = str(
        importlib_resources.files(PACKAGE_NAME).joinpath(FLOWS_SCHEMA_FILE)
    )
    # Deepcopy before mutating so we never alter the reader result (e.g. if cached).
    flows_schema = copy.deepcopy(read_json_file(flows_schema_file_path))
    if not isinstance(flows_schema, dict):
        raise ValueError("Flows schema file is not a dictionary")
    flows_schema = cast(Dict[str, Any], flows_schema)
    if calm_only:
        _remove_nlu_trigger_from_flows_schema(flows_schema)
    return flows_schema


def _remove_nlu_trigger_from_flows_schema(flows_schema: Dict[str, Any]) -> None:
    """Remove nlu_trigger from the flow definition and from $defs if present."""
    try:
        flows_schema["$defs"]["flow"]["properties"].pop("nlu_trigger", None)
    except Exception as e:
        raise ValueError("Flows schema does not have expected structure") from e


def get_e2e_tests_schema() -> dict[str, Any]:
    # Deepcopy to avoid mutating the LRU-cached schema
    # (read_schema_file → read_yaml_file).
    e2e_tests_schema = copy.deepcopy(
        read_schema_file(E2E_TESTS_SCHEMA_FILE, PACKAGE_NAME, False)
    )
    if not isinstance(e2e_tests_schema, dict):
        raise ValueError("E2E tests schema file is not a dictionary")
    return e2e_tests_schema


def get_domain_responses_schema() -> dict[str, Any]:
    # Deepcopy to avoid mutating the LRU-cached schema
    # (read_schema_file → read_yaml_file).
    responses_schema = copy.deepcopy(
        read_schema_file(RESPONSES_SCHEMA_FILE, PACKAGE_NAME, False)
    )
    if not isinstance(responses_schema, dict):
        raise ValueError("Responses schema file is not a dictionary")
    responses_schema_raw = cast(
        Optional[Dict[str, Any]],
        responses_schema.get("schema;responses"),
    )
    if responses_schema_raw is None:
        raise ValueError("Responses schema is not a dictionary")
    return responses_schema_raw


def get_domain_schema(
    include_responses: bool = True,
    calm_only: bool = True,
) -> dict[str, Any]:
    # Deepcopy to avoid mutating the LRU-cached schema
    # (read_schema_file → read_yaml_file).
    domain_schema = _get_domain_schema()

    # Originally, the domain schema does not include the responses section.
    if include_responses:
        responses_schema = get_domain_responses_schema()
        domain_schema["mapping"]["responses"] = responses_schema

    # Originally, the domain schema includes non-CALM properties.
    if calm_only:
        _remove_non_calm_domain_schema_keys(domain_schema)
        _remove_non_calm_slot_mappings(domain_schema)

    return domain_schema


def _get_domain_schema() -> dict[str, Any]:
    # Deepcopy to avoid mutating the LRU-cached schema
    # (read_schema_file → read_yaml_file).
    domain_schema = copy.deepcopy(
        read_schema_file(DOMAIN_SCHEMA_FILE, PACKAGE_NAME, False)
    )
    if not isinstance(domain_schema, dict):
        raise ValueError("Domain schema file is not a dictionary")
    return domain_schema


def _remove_non_calm_domain_schema_keys(domain_schema: Dict[str, Any]) -> None:
    """Remove the given top-level keys from the domain schema's mapping section.

    The domain schema has a "mapping" section that defines the structure of each
    top-level domain key (slots, actions, responses, etc.). This function removes
    the listed keys from that section in place.

    Args:
        domain_schema: The full domain schema dict (will be mutated).
        keys: Keys to remove from the mapping section (e.g. intents, entities).

    Raises:
        ValueError: If the schema has no "mapping" or it is not a dictionary.
    """
    # "mapping" defines each top-level domain key (slots, actions, responses, …)
    mapping_section = domain_schema.get("mapping")

    if not isinstance(mapping_section, dict):
        raise ValueError("Domain schema mapping is not a dictionary")

    mapping_section = cast(Dict[str, Any], mapping_section)

    for key in NON_CALM_DOMAIN_KEYS:
        mapping_section.pop(key, None)


def _remove_non_calm_slot_mappings(domain_schema: Dict[str, Any]) -> None:
    """Remove from the slot schema only properties that are not CALM-related.

    Keeps the full 'mappings' and 'validation' sections. From each mapping entry
    schema, removes: intent, not_intent, entity, role, group, value, action.
    From each condition entry schema, removes: active_loop, requested_slot.
    """
    try:
        slot_schema = domain_schema["mapping"]["slots"]["mapping"]["regex;([A-Za-z]+)"][
            "mapping"
        ]
        # Get all available slot `mappings` key
        slot_mappings_property_schema = slot_schema["mappings"]["sequence"][0][
            "mapping"
        ]
        # Get all available conditions for a slot mapping `conditions` key
        condition_mapping_schema = slot_mappings_property_schema["conditions"][
            "sequence"
        ][0]["mapping"]
    except Exception as e:
        raise ValueError(
            "Domain schema does not have expected slot mapping structure"
        ) from e

    # Remove all non-CALM properties from the slot mappings property schema
    for key in NON_CALM_SLOT_MAPPING_KEYS:
        slot_mappings_property_schema.pop(key, None)
    # Remove all non-CALM properties from the condition mapping schema
    for key in NON_CALM_CONDITION_KEYS:
        condition_mapping_schema.pop(key, None)
