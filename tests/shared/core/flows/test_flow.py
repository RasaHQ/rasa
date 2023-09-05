import pytest

from rasa.shared.core.flows.flow import (
    ElseFlowLink,
    IfFlowLink,
    build_flow_steps_list,
    END_STEP,
)


def test_build_flow_steps_list() -> None:
    steps = [
        {"id": "step1", "action": "action_search_hotel"},
        {"id": "step2", "action": "utter_hotel_inform_rating"},
    ]

    actual_steps = build_flow_steps_list(steps)

    assert actual_steps[0].id == "step1"
    assert actual_steps[0].next.links[0].target == "step2"
    assert actual_steps[0].action == "action_search_hotel"

    assert actual_steps[1].id == "step2"
    assert actual_steps[1].next.links[0].target == END_STEP
    assert actual_steps[1].action == "utter_hotel_inform_rating"


@pytest.mark.parametrize(
    "then_node, else_node, expected_if_target, expected_else_target",
    [
        ("confirm_booking", "END", "confirm_booking", END_STEP),
        ("END", "confirm_booking", END_STEP, "confirm_booking"),
    ],
)
def test_build_flow_steps_list_with_conditional_branching(
    then_node: str,
    else_node: str,
    expected_if_target: str,
    expected_else_target: str,
) -> None:
    steps = [
        {"id": "step1", "action": "action_search_hotel"},
        {
            "id": "step2",
            "action": "utter_hotel_inform_rating",
            "next": [{"if": "rating_above_4", "then": then_node}, {"else": else_node}],
        },
    ]

    actual_steps = build_flow_steps_list(steps)

    assert actual_steps[0].id == "step1"
    assert actual_steps[0].action == "action_search_hotel"
    assert actual_steps[0].next.links[0].target == "step2"

    assert actual_steps[1].id == "step2"
    assert actual_steps[1].action == "utter_hotel_inform_rating"
    assert len(actual_steps[1].next.links) == 2

    assert isinstance(actual_steps[1].next.links[0], IfFlowLink)
    assert actual_steps[1].next.links[0].target == expected_if_target

    assert isinstance(actual_steps[1].next.links[1], ElseFlowLink)
    assert actual_steps[1].next.links[1].target == expected_else_target


def test_build_flow_steps_list_with_next_as_end() -> None:
    steps = [
        {"id": "step1", "action": "action_search_hotel"},
        {
            "id": "step2",
            "action": "utter_hotel_inform_rating",
            "next": [{"if": "rating_above_4", "then": "step3"}, {"else": "step4"}],
        },
        {"id": "step3", "action": "utter_suggestion", "next": "END"},
        {"id": "step4", "action": "utter_no_suggestion", "next": "END"},
    ]

    actual_steps = build_flow_steps_list(steps)

    assert actual_steps[0].id == "step1"
    assert actual_steps[0].action == "action_search_hotel"
    assert actual_steps[0].next.links[0].target == "step2"

    assert actual_steps[1].id == "step2"
    assert actual_steps[1].action == "utter_hotel_inform_rating"
    assert len(actual_steps[1].next.links) == 2

    assert isinstance(actual_steps[1].next.links[0], IfFlowLink)
    assert actual_steps[1].next.links[0].target == "step3"

    assert isinstance(actual_steps[1].next.links[1], ElseFlowLink)
    assert actual_steps[1].next.links[1].target == "step4"

    assert actual_steps[2].id == "step3"
    assert actual_steps[2].action == "utter_suggestion"
    assert actual_steps[2].next.links[0].target == END_STEP

    assert actual_steps[3].id == "step4"
    assert actual_steps[3].action == "utter_no_suggestion"
    assert actual_steps[3].next.links[0].target == END_STEP
