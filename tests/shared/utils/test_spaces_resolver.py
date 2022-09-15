import pytest

from rasa.shared.core.constants import DEFAULT_INTENTS
from rasa.shared.utils.yaml_spaces_reader import YAMLSpacesReader
from rasa.shared.utils.space_resolver import SpaceResolver


def test_simple_resolving():
    spaces_yaml_path = "data/test_spaces/greetings_money_spaces.yml"
    spaces = YAMLSpacesReader.read_from_file(spaces_yaml_path)
    domain, training_data = SpaceResolver.resolve_spaces(spaces)

    expected_intents = {"greet", "goodbye", "money!send_money", "money!receive_money"}

    assert set(training_data.intents) == expected_intents

    assert set(domain.intents) == expected_intents.union(set(DEFAULT_INTENTS))