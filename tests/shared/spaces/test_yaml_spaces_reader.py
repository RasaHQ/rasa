import pytest

from rasa.shared.exceptions import YamlException
from rasa.shared.spaces.yaml_spaces_reader import YAMLSpacesReader


def test_yaml_spaces_reader():
    spaces_file = "data/test_spaces/first_spaces.yml"
    spaces = YAMLSpacesReader.read_from_file(spaces_file)

    assert len(spaces) == 2
    assert {s.name for s in spaces} == {"main", "transfer_money"}


def test_yaml_spaces_reader_validates_properly():
    spaces_file = "data/test_spaces/first_spaces_invalid.yml"
    with pytest.raises(YamlException):
        YAMLSpacesReader.read_from_file(spaces_file)
