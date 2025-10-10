from typing import Any, Dict, List

from rasa.shared.utils.yaml import read_yaml_file


async def test_alphabetical_order_default_responses() -> None:
    default_flows_responses = read_yaml_file(
        "rasa/dialogue_understanding/patterns/default_flows_for_patterns.yml"
    )
    responses = list(default_flows_responses["responses"].keys())
    # To make the test pass, add new responses to the default_flows_for_patterns.yml
    # in alphabetical order.
    assert responses == sorted(responses)


async def test_alphabetical_order_default_flows() -> None:
    default_flows_responses = read_yaml_file(
        "rasa/dialogue_understanding/patterns/default_flows_for_patterns.yml"
    )
    default_flows = list(default_flows_responses["flows"].keys())
    # To make the test pass, add new patterns to the default_flows_for_patterns.yml
    # in alphabetical order.
    assert default_flows == sorted(default_flows)


def _extract_conditions_from_flow_steps(steps: List[Dict[str, Any]]) -> List[str]:
    """Extract all condition strings from flow steps recursively."""
    conditions = []

    for step in steps:
        # Check for direct 'if' conditions
        if "if" in step:
            conditions.append(step["if"])

        # Check for conditions in 'next' arrays
        if "next" in step:
            next_steps = step["next"]
            if isinstance(next_steps, list):
                for next_step in next_steps:
                    if isinstance(next_step, dict) and "if" in next_step:
                        conditions.append(next_step["if"])
                    # Recursively check nested steps
                    if isinstance(next_step, dict) and "then" in next_step:
                        if isinstance(next_step["then"], list):
                            conditions.extend(
                                _extract_conditions_from_flow_steps(next_step["then"])
                            )
                        elif isinstance(next_step["then"], dict):
                            conditions.extend(
                                _extract_conditions_from_flow_steps([next_step["then"]])
                            )

    return conditions


async def test_default_patterns_do_not_use_inequality_operator() -> None:
    """Test that default patterns do not use '!=' operator for studio compatibility."""
    default_flows_responses = read_yaml_file(
        "rasa/dialogue_understanding/patterns/default_flows_for_patterns.yml"
    )

    flows = default_flows_responses["flows"]
    violations = []

    for flow_name, flow_data in flows.items():
        if "steps" in flow_data:
            conditions = _extract_conditions_from_flow_steps(flow_data["steps"])

            for condition in conditions:
                if isinstance(condition, str) and "!=" in condition:
                    violations.append(
                        f"Flow '{flow_name}' uses '!=' operator in "
                        f"condition: '{condition}'"
                    )

    assert len(violations) == 0


async def test_inequality_operator_violation_detection() -> None:
    """Test that the inequality operator test correctly detects violations."""
    # Create a mock flow structure with != operator to test violation detection
    mock_flows_with_violations = {
        "flows": {
            "test_pattern_with_inequality": {
                "description": "Test pattern with inequatity operator",
                "steps": [
                    {
                        "noop": True,
                        "next": [
                            {
                                # This should trigger violation
                                "if": "slots.test_slot != 'value'",
                                "then": [{"action": "utter_test"}],
                            }
                        ],
                    }
                ],
            }
        }
    }

    violations = []

    for flow_name, flow_data in mock_flows_with_violations["flows"].items():
        if "steps" in flow_data:
            conditions = _extract_conditions_from_flow_steps(flow_data["steps"])

            for condition in conditions:
                if isinstance(condition, str) and "!=" in condition:
                    violations.append(
                        f"Flow '{flow_name}' uses '!=' operator in "
                        f"condition: '{condition}'"
                    )

    # Verify that violations are detected
    assert (
        len(violations) == 1
    ), f"Expected 1 violation, but found {len(violations)}: {violations}"

    # Verify specific violation is detected
    assert "slots.test_slot != 'value'" in violations[0]
