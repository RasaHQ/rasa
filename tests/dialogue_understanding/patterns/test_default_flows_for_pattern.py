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
