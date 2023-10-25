import pytest

from rasa.shared.core.flows.flow import Flow, FlowsList
from rasa.shared.importers.importer import FlowSyncImporter
from tests.utilities import flows_from_str


@pytest.fixture
def user_flows_and_patterns() -> FlowsList:
    return flows_from_str(
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


@pytest.fixture
def only_patterns() -> FlowsList:
    return flows_from_str(
        """
        flows:
          pattern_bar:
            steps:
            - id: first
              action: action_listen
        """
    )


@pytest.fixture
def empty_flowlist() -> FlowsList:
    return FlowsList(flows=[])


def test_user_flow_ids(user_flows_and_patterns: FlowsList):
    assert user_flows_and_patterns.user_flow_ids == ["foo"]


def test_user_flow_ids_handles_empty(empty_flowlist: FlowsList):
    assert empty_flowlist.user_flow_ids == []


def test_user_flow_ids_handles_patterns_only(only_patterns: FlowsList):
    assert only_patterns.user_flow_ids == []


def test_user_flows(user_flows_and_patterns: FlowsList):
    user_flows = user_flows_and_patterns.user_flows
    expected_user_flows = FlowsList(
        [Flow.from_json("foo", {"steps": [{"id": "first", "action": "action_listen"}]})]
    )
    assert user_flows == expected_user_flows


def test_user_flows_handles_empty(empty_flowlist: FlowsList):
    assert empty_flowlist.user_flows == empty_flowlist


def test_user_flows_handles_patterns_only(
    only_patterns: FlowsList, empty_flowlist: FlowsList
):
    assert only_patterns.user_flows == empty_flowlist


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
