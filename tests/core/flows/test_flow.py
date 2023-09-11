from rasa.shared.core.flows.flow import FlowsList
from tests.utilities import flows_from_yaml


def test_non_pattern_flows():
    all_flows = flows_from_yaml(
        """
        flows:
          foo:
            steps:
              - id: first
                action: action_listen
          pattern_bar:
            steps:
            - id: first
              action: action_listen
        """
    )
    assert all_flows.non_pattern_flows() == ["foo"]


def test_non_pattern_handles_empty():
    assert FlowsList(flows=[]).non_pattern_flows() == []


def test_non_pattern_flows_handles_patterns_only():
    all_flows = flows_from_yaml(
        """
        flows:
          pattern_bar:
            steps:
            - id: first
              action: action_listen
        """
    )
    assert all_flows.non_pattern_flows() == []
