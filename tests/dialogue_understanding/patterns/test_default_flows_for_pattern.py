import rasa.shared.utils.io


async def test_alphabetical_order_default_responses() -> None:
    default_flows_responses = rasa.shared.utils.io.read_yaml_file(
        "rasa/dialogue_understanding/patterns/default_flows_for_patterns.yml"
    )
    response = default_flows_responses["responses"]
    expected_responses = [
        "utter_can_do_something_else",
        "utter_clarification_options_rasa",
        "utter_corrected_previous_input",
        "utter_flow_cancelled_rasa",
        "utter_flow_continue_interrupted",
        "utter_inform_code_change",
        "utter_internal_error_rasa",
        "utter_no_knowledge_base",
    ]
    # To make the test pass, add new responses to the default_flows_for_patterns.yml
    # in alphabetical order.
    assert list(response.keys()) == sorted(expected_responses)


async def test_alphabetical_order_default_flows() -> None:
    default_flows_responses = rasa.shared.utils.io.read_yaml_file(
        "rasa/dialogue_understanding/patterns/default_flows_for_patterns.yml"
    )
    default_flows = default_flows_responses["flows"]
    expected_flows = [
        "pattern_cancel_flow",
        "pattern_chitchat",
        "pattern_clarification",
        "pattern_code_change",
        "pattern_collect_information",
        "pattern_completed",
        "pattern_continue_interrupted",
        "pattern_correction",
        "pattern_internal_error",
        "pattern_search",
    ]
    # To make the test pass, add new pattens to the default_flows_for_patterns.yml
    # in alphabetical order.
    assert list(default_flows.keys()) == sorted(expected_flows)
