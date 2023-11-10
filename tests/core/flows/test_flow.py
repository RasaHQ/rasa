from typing import Tuple

import pytest

from rasa.shared.core.flows import Flow, FlowsList
from rasa.shared.core.flows.flow_step_links import StaticFlowStepLink
from rasa.shared.core.flows.steps import (
    ActionFlowStep,
    ContinueFlowStep,
    EndFlowStep,
    StartFlowStep,
)
from rasa.shared.core.flows.steps.constants import (
    CONTINUE_STEP_PREFIX,
    END_STEP,
    START_STEP,
)
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
def add_contact_flow() -> Flow:
    flows = flows_from_str(
        """
            flows:
              add_contact:
                description: add a contact to your contact list
                name: add a contact
                steps:
                  - collect: "add_contact_handle"
                    description: "a user handle starting with @"
                  - collect: "add_contact_name"
                    description: "a name of a person"
                  - collect: "add_contact_confirmation"
                    ask_before_filling: true
                    next:
                      - if: not add_contact_confirmation
                        then:
                          - action: utter_add_contact_cancelled
                            next: END
                      - else: add_contact
                  - id: add_contact
                    action: add_contact
                    next:
                      - if: "return_value = 'success'"
                        then:
                          - action: utter_contact_added
                            next: END
                      - if: "return_value = 'already_exists'"
                        then:
                          - action: utter_contact_already_exists
                            next: END
                      - else:
                          - action: utter_add_contact_error
                            next: END
            """
    )
    add_contact_flow = flows.flow_by_id("add_contact")
    assert add_contact_flow is not None
    return add_contact_flow


add_contact_flow_collects = (
    "add_contact_handle",
    "add_contact_name",
    "add_contact_confirmation",
)


@pytest.fixture
def empty_flows_list() -> FlowsList:
    return FlowsList(underlying_flows=[])


def test_user_flow_ids(user_flows_and_patterns: FlowsList):
    assert user_flows_and_patterns.user_flow_ids == {"foo"}


def test_user_flow_ids_handles_empty(empty_flows_list: FlowsList):
    assert empty_flows_list.user_flow_ids == set()


def test_user_flow_ids_handles_patterns_only(only_patterns: FlowsList):
    assert only_patterns.user_flow_ids == set()


def test_user_flows(user_flows_and_patterns: FlowsList):
    user_flows = user_flows_and_patterns.user_flows
    expected_user_flows = FlowsList(
        [Flow.from_json("foo", {"steps": [{"id": "first", "action": "action_listen"}]})]
    )
    assert user_flows == expected_user_flows


def test_user_flows_handles_empty(empty_flows_list: FlowsList):
    assert empty_flows_list.user_flows == empty_flows_list


def test_user_flows_handles_patterns_only(
    only_patterns: FlowsList, empty_flows_list: FlowsList
):
    assert only_patterns.user_flows == empty_flows_list


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


def test_flows_list_length(user_flows_and_patterns: FlowsList):
    assert len(user_flows_and_patterns) == 2


def test_flows_list_is_empty(user_flows_and_patterns: FlowsList):
    assert not user_flows_and_patterns.is_empty()
    assert FlowsList(underlying_flows=[]).is_empty()


def test_flows_list_as_json_list(user_flows_and_patterns: FlowsList):
    json_list = user_flows_and_patterns.as_json_list()
    assert len(json_list) == len(user_flows_and_patterns)
    assert set([f["id"] for f in json_list]) == user_flows_and_patterns.flow_ids


def test_fingerprinting_flows_list_returns_str(user_flows_and_patterns: FlowsList):
    assert isinstance(user_flows_and_patterns.fingerprint(), str)


def test_fingerprinting_flows_list_is_stable(user_flows_and_patterns: FlowsList):
    fp_1 = user_flows_and_patterns.fingerprint()
    fp_2 = user_flows_and_patterns.fingerprint()
    assert fp_1 == fp_2


def test_fingerprinting_flows_list_creates_distinct_fingerprints(
    user_flows_and_patterns: FlowsList,
):
    only_one_flow = FlowsList(user_flows_and_patterns.underlying_flows[:1])
    assert only_one_flow.fingerprint() != user_flows_and_patterns.fingerprint()


def test_merging_flows_lists(user_flows_and_patterns: FlowsList):
    additional_flows = flows_from_str(
        """
        flows:
          abc_x:
            description: an additional flow
            steps:
              - action: utter_greet
    """
    )
    merged_flows = user_flows_and_patterns.merge(additional_flows)
    assert len(merged_flows) == len(user_flows_and_patterns) + 1
    assert (
        user_flows_and_patterns.flow_ids | additional_flows.flow_ids
        == merged_flows.flow_ids
    )


def test_flow_by_id_returns_none_for_non_existing_flow(
    user_flows_and_patterns: FlowsList,
):
    assert user_flows_and_patterns.flow_by_id("XXX") is None


def test_flow_by_id_returns_correct_flow(user_flows_and_patterns: FlowsList):
    flow_id = "foo"
    retrieved_flow = user_flows_and_patterns.flow_by_id(flow_id)
    assert retrieved_flow is not None
    assert retrieved_flow.id == flow_id


def test_create_default_name(user_flows_and_patterns: FlowsList):
    assert {"foo", "pattern bar"} == {f.name for f in user_flows_and_patterns}


def test_assignment_of_default_ids(add_contact_flow: Flow):
    assert [s.default_id for s in add_contact_flow.steps] == [
        "0_collect_add_contact_handle",
        "1_collect_add_contact_name",
        "2_collect_add_contact_confirmation",
        "3_utter_add_contact_cancelled",
        "4_add_contact",
        "5_utter_contact_added",
        "6_utter_contact_already_exists",
        "7_utter_add_contact_error",
    ]


def test_step_by_id_returns_none_for_non_existing_id(add_contact_flow: Flow):
    assert add_contact_flow.step_by_id("XXXXX@XXXXX") is None


def test_step_by_id_returns_the_right_step(add_contact_flow: Flow):
    step_id = "add_contact"
    retrieved_step = add_contact_flow.step_by_id(step_id)
    assert retrieved_step.id == step_id
    assert isinstance(retrieved_step, ActionFlowStep)
    assert retrieved_step.action == "add_contact"


def test_step_by_id_returns_proper_start_step(add_contact_flow: Flow):
    start_step = add_contact_flow.step_by_id(START_STEP)
    assert start_step is not None
    assert isinstance(start_step, StartFlowStep)
    assert len(start_step.next.links) == 1
    link = start_step.next.links[0]
    assert isinstance(link, StaticFlowStepLink)
    assert link.target == "0_collect_add_contact_handle"


def test_step_by_id_returns_end_step(add_contact_flow: Flow):
    end_step = add_contact_flow.step_by_id(END_STEP)
    assert end_step is not None
    assert isinstance(end_step, EndFlowStep)


def test_step_by_id_returns_proper_continuation_step(add_contact_flow: Flow):
    target_id = "0_collect_add_contact_handle"
    continuation_step_id = f"{CONTINUE_STEP_PREFIX}{target_id}"
    continuation_step = add_contact_flow.step_by_id(continuation_step_id)
    assert continuation_step is not None
    assert isinstance(continuation_step, ContinueFlowStep)
    assert len(continuation_step.next.links) == 1
    link = continuation_step.next.links[0]
    assert isinstance(link, StaticFlowStepLink)
    assert link.target == target_id


def test_first_step_in_flow(add_contact_flow: Flow):
    assert add_contact_flow.first_step_in_flow().id == "0_collect_add_contact_handle"


def test_is_rasa_default_flow(user_flows_and_patterns: FlowsList):
    foo_flow = user_flows_and_patterns.flow_by_id("foo")
    assert foo_flow is not None
    assert not foo_flow.is_rasa_default_flow

    pattern_bar_flow = user_flows_and_patterns.flow_by_id("pattern_bar")
    assert pattern_bar_flow is not None
    assert pattern_bar_flow.is_rasa_default_flow


def test_get_collect_steps(add_contact_flow: Flow):
    collect_steps = add_contact_flow.get_collect_steps()
    assert len(collect_steps) == 3
    assert set(add_contact_flow_collects) == {s.collect for s in collect_steps}


@pytest.mark.parametrize(
    "step_id, expected_collects",
    [
        ("0_collect_add_contact_handle", add_contact_flow_collects[:1]),
        ("1_collect_add_contact_name", add_contact_flow_collects[:2]),
        ("2_collect_add_contact_confirmation", add_contact_flow_collects[:3]),
    ],
)
def test_basic_previous_collect_steps(
    step_id: str, expected_collects: Tuple[str, ...], add_contact_flow: Flow
):
    collects = add_contact_flow.previous_collect_steps(step_id)
    assert [s.collect for s in collects] == list(reversed(expected_collects))


def test_flow_step_iteration_in_deeply_nested_flow():
    flows = flows_from_str(
        """
            flows:
              deep_flow:
                steps:
                  - collect: x
                    next:
                      - if: x is even
                        then:
                          - collect: y
                            next:
                              - if: y is even
                                then:
                                  - collect: z
                                    next:
                                      - if: z is even
                                        then: win
                                      - else:
                                        - action: utter_too_bad
                                        - action: utter_so_close
                                          next: lose
                              - else: lose
                      - else: lose
                  - id: win
                    action: utter_win
                    next: END
                  - id: lose
                    action: utter_lose
                    next: END
            """
    )
    flow = flows.flow_by_id("deep_flow")
    assert flow is not None
    assert len(flow.steps) == 7


@pytest.mark.parametrize(
    "guard_condition, expected_startable",
    [
        ("True", True),
        ("False", False),
        ("True and False", False),
        ("True or False", True),
        ("context.x > 0", True),
        ("context.x < 0", False),
        ("slots.spam is not null", True),
        ("slots.spam is 'eggs'", True),
        ("slots.spam is 'ham'", False),
        ("slots.authenticated AND slots.email_verified", True),
        ("slots.some_missing_slot is 'available'", False),
    ],
)
def test_is_startable(guard_condition: str, expected_startable: bool):
    """Test that the start condition is evaluated correctly."""
    # Given
    flow = Flow.from_json(
        "foo",
        {"if": guard_condition, "steps": [{"id": "first", "action": "action_listen"}]},
    )
    document = {
        "context": {"x": 2},
        "slots": {"spam": "eggs", "authenticated": True, "email_verified": True},
    }
    # When
    is_startable = flow.is_startable(document)
    # Then
    assert is_startable == expected_startable


@pytest.mark.parametrize(
    "nlu_trigger_config, actual_intents",
    [
        (
            """nlu_trigger:
                 - intent: bar""",
            ["bar"],
        ),
        (
            """nlu_trigger:
                 - intent: bar
                 - intent: foo""",
            ["bar", "foo"],
        ),
        (
            "",
            [],
        ),
    ],
)
def test_get_trigger_intents(nlu_trigger_config, actual_intents):
    flows = flows_from_str(
        f"""
        flows:
          foo:
            {nlu_trigger_config}
            steps:
              - action: utter_welcome
        """
    )

    intents = flows.underlying_flows[0].get_trigger_intents()

    assert intents == set(actual_intents)
