"""Tests for exit_if condition validation in flows."""

import textwrap
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from rasa.core.available_agents import AvailableAgents
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows.validation import InvalidExitIfConditionException
from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader


@pytest.fixture
def mock_available_agents(monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    """Mock available agents for testing."""
    mock_instance = MagicMock()
    mock_instance.agents = {
        "some_agent": {},
        "car-research": {},
    }

    with patch.object(
        AvailableAgents, "get_instance", return_value=mock_instance
    ) as mock_method:
        yield mock_method


@pytest.mark.parametrize(
    "condition,required_slots",
    [
        ("slots.x > 0", {"x": {"type": "float"}}),
        ("slots.age >= 18", {"age": {"type": "float"}}),
        ("slots.status == 'completed'", {"status": {"type": "text"}}),
        ("slots.is_active is True", {"is_active": {"type": "bool"}}),
        (
            "slots.count is not null",
            {"count": {"type": "float"}},
        ),  # Use float instead of int
    ],
)
def test_validate_exit_if_conditions_valid(
    mock_available_agents: MagicMock, condition: str, required_slots: dict
):
    """Test that valid exit_if conditions pass validation."""
    data = textwrap.dedent(
        f"""
        flows:
          my_flow:
            description: test flow with valid exit_if
            steps:
              - call: some_agent
                exit_if:
                  - {condition}
        """
    )

    flows = YAMLFlowsReader.read_from_string(data)

    # Create domain with the required slots
    domain_data = {"slots": required_slots}
    domain = Domain.from_dict(domain_data)

    # Should not raise any exception
    flows.validate(domain)


@pytest.mark.parametrize(
    "conditions,slots",
    [
        (
            ["slots.x > 0 and slots.y < 10", "slots.z == 'done'"],
            {"x": {"type": "float"}, "y": {"type": "float"}, "z": {"type": "text"}},
        ),
        (
            [
                (
                    "(slots.status == 'completed' and slots.score > 80)"
                    " or slots.force_exit"
                )
            ],
            {
                "status": {"type": "text"},
                "score": {"type": "float"},
                "force_exit": {"type": "bool"},
            },
        ),
    ],
)
def test_validate_exit_if_conditions_multiple_slots(
    mock_available_agents: MagicMock, conditions: list, slots: dict
):
    """Test that exit_if conditions with multiple slots are validated correctly."""
    # Create the YAML with proper indentation
    conditions_yaml = "\n".join(
        f"                      - {condition}" for condition in conditions
    )
    data = textwrap.dedent(
        f"""
        flows:
          my_flow:
            description: test flow with multiple slots
            steps:
              - call: some_agent
                exit_if:
{conditions_yaml}
        """
    )

    flows = YAMLFlowsReader.read_from_string(data)

    # Create domain with all referenced slots
    domain_data = {"slots": slots}
    domain = Domain.from_dict(domain_data)

    # Should not raise any exception
    flows.validate(domain)


@pytest.mark.parametrize(
    "condition,slot_type",
    [
        ("slots.boolean_slot == True", "bool"),
        ("slots.boolean_slot != False", "bool"),
        ("slots.boolean_slot is not null", "bool"),
        ("slots.numeric_slot > 10", "float"),
        ("slots.count <= 42", "float"),
        ("slots.text_slot == 'value'", "text"),
        ("slots.categorical_slot != 'option'", "categorical"),
        ("slots.list_slot is null", "list"),
        ("slots.any_slot == 123", "any"),
        ("slots.any_slot == True", "any"),
        ("slots.any_slot == 'text'", "any"),
    ],
)
def test_validate_exit_if_conditions_valid_syntax(
    mock_available_agents: MagicMock, condition: str, slot_type: str
):
    """Test that exit_if conditions with valid syntax pass validation.

    This test verifies that various valid exit_if conditions with different
    slot types pass validation without errors. It ensures our validation
    pipeline doesn't incorrectly reject valid conditions.
    """
    data = textwrap.dedent(
        f"""
        flows:
          my_flow:
            description: test flow with valid syntax
            steps:
              - call: some_agent
                exit_if:
                  - {condition}
        """
    )

    flows = YAMLFlowsReader.read_from_string(data)

    # Create domain with the slot defined with the specific type
    slot_name = condition.split(".")[1].split()[0]  # Extract slot name from condition
    domain_data = {"slots": {slot_name: {"type": slot_type}}}
    domain = Domain.from_dict(domain_data)

    # Should not raise any exception
    flows.validate(domain)


@pytest.mark.parametrize(
    "condition,expected_error",
    [
        (
            "recommendation_what_stock_to_buy is not null",
            "must contain at least one slot reference",
        ),
        (
            "status == 'completed'",
            "must contain at least one slot reference",
        ),
        (
            "score > 80",
            "must contain at least one slot reference",
        ),
    ],
)
def test_validate_exit_if_conditions_missing_slots_prefix(
    mock_available_agents: MagicMock, condition: str, expected_error: str
):
    """Test that exit_if conditions without slots. prefix are rejected."""
    data = textwrap.dedent(
        f"""
        flows:
          my_flow:
            description: test flow with invalid exit_if
            steps:
              - call: some_agent
                exit_if:
                  - {condition}
        """
    )

    flows = YAMLFlowsReader.read_from_string(data)
    domain = Domain.empty()

    with pytest.raises(InvalidExitIfConditionException) as exc_info:
        flows.validate(domain)

    assert expected_error in str(exc_info.value)
    assert condition in str(exc_info.value)


@pytest.mark.parametrize(
    "condition,slot_name,expected_error",
    [
        (
            "slots.undefined_slot > 0",
            "undefined_slot",
            "Slot 'undefined_slot' is not defined in the domain",
        ),
        (
            "slots.missing_slot == 'value'",
            "missing_slot",
            "Slot 'missing_slot' is not defined in the domain",
        ),
        (
            "slots.unknown_slot is not null",
            "unknown_slot",
            "Slot 'unknown_slot' is not defined in the domain",
        ),
    ],
)
def test_validate_exit_if_conditions_undefined_slot(
    mock_available_agents: MagicMock,
    condition: str,
    slot_name: str,
    expected_error: str,
):
    """Test that exit_if conditions with undefined slots are rejected."""
    data = textwrap.dedent(
        f"""
        flows:
          my_flow:
            description: test flow with undefined slot
            steps:
              - call: some_agent
                exit_if:
                  - {condition}
        """
    )

    flows = YAMLFlowsReader.read_from_string(data)

    # Create domain with some slots but not the undefined one
    domain_data = {
        "slots": {
            "defined_slot": {"type": "float"},
            "another_slot": {"type": "text"},
        }
    }
    domain = Domain.from_dict(domain_data)

    with pytest.raises(InvalidExitIfConditionException) as exc_info:
        flows.validate(domain)

    assert expected_error in str(exc_info.value)


@pytest.mark.parametrize(
    "condition,expected_error",
    [
        ("slots.x > > 18", "Invalid predicate"),
        ("slots.y == == 'value'", "Invalid predicate"),
        ("slots.z is is not null", "Invalid predicate"),
    ],
)
def test_validate_exit_if_conditions_invalid_predicate(
    mock_available_agents: MagicMock, condition: str, expected_error: str
):
    """Test that exit_if conditions with invalid predicates are rejected."""
    data = textwrap.dedent(
        f"""
        flows:
          my_flow:
            description: test flow with invalid predicate
            steps:
              - call: some_agent
                exit_if:
                  - {condition}
        """
    )

    flows = YAMLFlowsReader.read_from_string(data)

    # Create domain with the slot defined so we can test predicate validation
    domain_data = {
        "slots": {
            "x": {"type": "float"},
            "y": {"type": "text"},
            "z": {"type": "text"},
        }
    }
    domain = Domain.from_dict(domain_data)

    with pytest.raises(InvalidExitIfConditionException) as exec_info:
        flows.validate(domain)

    assert expected_error in str(exec_info.value)


def test_validate_exit_if_conditions_non_string_condition(
    mock_available_agents: MagicMock,
):
    """Test that non-string exit_if conditions are rejected."""
    data = textwrap.dedent(
        """
        flows:
          my_flow:
            description: test flow with non-string condition
            steps:
              - call: some_agent
                exit_if:
                  - "slots.x > 0"
        """
    )

    flows = YAMLFlowsReader.read_from_string(data)

    # Manually modify the condition to be non-string after YAML parsing
    # This bypasses YAML schema validation to test our custom validation
    call_step = flows.flow_by_id("my_flow").steps[0]
    call_step.exit_if = [42]  # Non-string condition

    domain = Domain.empty()

    with pytest.raises(InvalidExitIfConditionException) as exc_info:
        flows.validate(domain)

    assert "Condition must be a string" in str(exc_info.value)


def test_validate_exit_if_conditions_without_domain(mock_available_agents: MagicMock):
    """Test that validation works without domain (skips slot name validation)."""
    data = textwrap.dedent(
        """
        flows:
          my_flow:
            description: test flow with exit_if
            steps:
              - call: some_agent
                exit_if:
                  - slots.undefined_slot > 0
        """
    )

    flows = YAMLFlowsReader.read_from_string(data)

    # Should not raise any exception when no domain is provided
    flows.validate()
