from rasa.shared.core.flows.flow import FlowsList
from rasa.shared.core.flows.yaml_flows_io import flows_from_str
from rasa.shared.importers.importer import FlowSyncImporter


def test_non_pattern_flows():
    all_flows = flows_from_str(
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
    all_flows = flows_from_str(
        """
        flows:
          pattern_bar:
            steps:
            - id: first
              action: action_listen
        """
    )
    assert all_flows.non_pattern_flows() == []


def test_collecting_flow_utterances():
    all_flows = flows_from_str(
        """
        flows:
          foo:
            steps:
              - action: utter_welcome
              - action: setup
              - collect: age
                rejections:
                  - if: age<18
                    utter: utter_too_young
                  - if: age>100
                    utter: utter_too_old
          bar:
            steps:
              - action: utter_hello
              - collect: income
                utter: utter_ask_income_politely
        """
    )
    assert all_flows.utterances == {
        "utter_ask_age",
        "utter_ask_income_politely",
        "utter_hello",
        "utter_welcome",
        "utter_too_young",
        "utter_too_old",
    }


def test_default_flows_have_non_empty_names():
    default_flows = FlowSyncImporter.load_default_pattern_flows()
    for flow in default_flows.underlying_flows:
        assert flow.name
