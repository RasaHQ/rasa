import os

from rasa.shared.core.constants import DEFAULT_INTENTS
from rasa.shared.core.domain import Domain
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.spaces.yaml_spaces_reader import YAMLSpacesReader
from rasa.shared.spaces.space_resolver import SpaceResolver


def test_simple_resolving():
    spaces_yaml_path = "data/test_spaces/greetings_money_spaces.yml"
    spaces = YAMLSpacesReader.read_from_file(spaces_yaml_path)
    path_with_joined_files = SpaceResolver.resolve_spaces(spaces)

    assert os.path.isdir(path_with_joined_files)

    domains_path = os.path.join(path_with_joined_files, "domain")
    nlu_path = os.path.join(path_with_joined_files, "nlu")

    assert os.path.isdir(domains_path)
    assert os.path.isdir(nlu_path)

    expected_intents = {"greet", "goodbye", "money!send_money", "money!receive_money"}

    domain = Domain.load(domains_path)
    assert set(domain.intents) == expected_intents.union(set(DEFAULT_INTENTS))

    training_data = RasaFileImporter(training_data_paths=nlu_path).get_nlu_data()

    assert set(training_data.intents) == expected_intents

