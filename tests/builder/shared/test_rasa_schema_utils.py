"""Tests for rasa_schema_utils.

One parametrized test per schema; use cases are the parameter sets.
Flows schema is JSON (JSON Schema); domain and e2e schemas are YAML.
"""

import jsonschema
import pytest

from rasa.builder.shared.rasa_schema_utils import (
    NON_CALM_CONDITION_KEYS,
    NON_CALM_DOMAIN_KEYS,
    NON_CALM_SLOT_MAPPING_KEYS,
    get_domain_schema,
    get_e2e_tests_schema,
    get_flows_schema,
)


@pytest.mark.parametrize(
    "calm_only",
    [True, False],
    ids=["calm_only", "full_schema"],
)
def test_get_flows_schema(calm_only: bool) -> None:
    """Test the get_flows_schema function."""
    # When
    schema = get_flows_schema(calm_only=calm_only)

    # Then
    assert isinstance(schema, dict)
    flow_props = schema["$defs"]["flow"]["properties"]
    if calm_only:
        assert "nlu_trigger" not in flow_props
    else:
        assert "nlu_trigger" in flow_props
    assert "steps" in flow_props and "description" in flow_props
    jsonschema.Draft7Validator.check_schema(schema)


@pytest.mark.parametrize(
    "use_case",
    ["valid_dict", "has_mapping", "has_fixtures_metadata_stub_actions"],
    ids=["valid_dict", "has_mapping", "required_mapping_keys"],
)
def test_get_e2e_tests_schema(use_case: str) -> None:
    """Single test for e2e tests schema; parametrized over use cases."""
    # When
    schema = get_e2e_tests_schema()

    # Then
    assert isinstance(schema, dict)


@pytest.mark.parametrize(
    "include_responses,calm_only",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
    ids=["responses_calm", "responses_full", "no_responses_calm", "no_responses_full"],
)
def test_get_domain_schema(include_responses: bool, calm_only: bool) -> None:
    """Single test for domain schema; parametrized over use cases."""
    # When
    schema = get_domain_schema(include_responses=include_responses, calm_only=calm_only)

    # Then
    assert isinstance(schema, dict)
    assert "mapping" in schema
    mapping = schema["mapping"]
    assert "slots" in mapping and "actions" in mapping and "responses" in mapping

    responses = mapping["responses"]
    slot_schema = mapping["slots"]["mapping"]["regex;([A-Za-z]+)"]["mapping"]
    entry = slot_schema["mappings"]["sequence"][0]["mapping"]
    cond = entry["conditions"]["sequence"][0]["mapping"]

    if include_responses:
        responses = mapping["responses"]
        assert isinstance(responses, dict) and (
            "mapping" in responses or "type" in responses
        )
        # Integrated responses are the raw schema (value of schema;responses from
        # responses.yml), not a dict keyed by "schema;responses".
        assert "schema;responses" not in responses
        if "mapping" in responses:
            assert any("regex" in k for k in responses["mapping"])  # type: ignore[union-attr]

    if calm_only:
        for key in NON_CALM_DOMAIN_KEYS:
            assert key not in mapping
        for key in NON_CALM_SLOT_MAPPING_KEYS:
            assert key not in entry
        for key in NON_CALM_CONDITION_KEYS:
            assert key not in cond
        assert "type" in entry and "coexistence_system" in entry
        assert "active_flow" in cond
    else:
        for key in NON_CALM_DOMAIN_KEYS:
            assert key in mapping
        for key in NON_CALM_SLOT_MAPPING_KEYS:
            assert key in entry
        for key in NON_CALM_CONDITION_KEYS:
            assert key in cond
