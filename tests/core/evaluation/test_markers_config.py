from rasa.core.evaluation.markers import MarkerConfig, InvalidMarkersConfig

import pytest
from typing import Text
from ruamel.yaml.parser import ParserError

from jsonschema import ValidationError


@pytest.fixture
def simple_marker_config_json() -> dict:
    """Returns a json dict equivalent to simple_marker_config fixture"""
    sample_json = {
        "markers": [
            {
                "marker": "carbon_offset_calculated",
                "operator": "AND",
                "condition": [
                    {
                        "slot_is_set": [
                            "travel_flight_class",
                            "travel_departure",
                            "travel_destination",
                        ]
                    },
                    {"action_executed": ["provide_carbon_estimate"]},
                ],
            },
        ]
    }
    return sample_json


@pytest.fixture
def multi_marker_config_json() -> dict:
    """Returns a json dict equivalent to multi_marker_config_folder fixture"""
    sample_json = {
        "markers": [
            {
                "marker": "no_restart",
                "condition": [{"action_not_executed": ["action_restart"]}],
            },
            {
                "marker": "all_required_data_gathered",
                "condition": [
                    {
                        "slot_is_set": [
                            "travel_flight_class",
                            "travel_departure",
                            "travel_destination",
                        ]
                    }
                ],
            },
            {
                "marker": "carbon_offset_calculated",
                "operator": "AND",
                "condition": [
                    {
                        "slot_is_set": [
                            "travel_flight_class",
                            "travel_departure",
                            "travel_destination",
                        ]
                    },
                    {"action_executed": ["provide_carbon_estimate"]},
                ],
            },
        ]
    }
    return sample_json


def test_empty_config():
    """Tests the format of an empty markers config."""
    assert MarkerConfig.empty_config() == {}


def test_from_yaml(simple_marker_config_json):
    """Tests the creation of a dict config from yaml string"""
    simple_yaml_markers_config = """
    markers:
      - marker: carbon_offset_calculated
        operator: AND
        condition:
          - slot_is_set:
            - travel_flight_class
            - travel_departure
            - travel_destination
          - action_executed:
            - provide_carbon_estimate
    """
    yaml_as_dict = MarkerConfig.from_yaml(simple_yaml_markers_config)
    assert yaml_as_dict == simple_marker_config_json


def test_invalid_yaml_exceptions(invalid_markers_config: Text):
    """Checks that an exception is raised when an invalid config file is supplied"""
    with pytest.raises(ParserError):
        MarkerConfig.from_file(invalid_markers_config)


def test_load_invalid_path():
    """Checks that an exception is raised when an invalid path is supplied"""
    with pytest.raises(InvalidMarkersConfig):
        MarkerConfig.load_config_from_path("not a path")


def test_load_valid_file(simple_marker_config: Text, simple_marker_config_json):
    """Tests the single config loader"""
    yaml_as_dict = MarkerConfig.load_config_from_path(simple_marker_config)
    assert yaml_as_dict == simple_marker_config_json


def test_load_valid_path(multi_marker_config_folder, multi_marker_config_json):
    """Tests the config folder loading"""
    yaml_as_dict = MarkerConfig.load_config_from_path(multi_marker_config_folder)
    # check that the two configs contain the same entries.
    for m in yaml_as_dict["markers"]:
        assert m in multi_marker_config_json["markers"]
    for m in multi_marker_config_json["markers"]:
        assert m in yaml_as_dict["markers"]


def test_valid_config(simple_marker_config_json):
    assert MarkerConfig._validate_config(simple_marker_config_json) is True


def test_config_missing_required_top_level_markers_label():
    sample_json = {"marker": "no_restart"}
    with pytest.raises(ValidationError):
        MarkerConfig._validate_config(sample_json)


def test_config_missing_required_marker_label():
    sample_json = {
        "markers": [{"condition": [{"action_not_executed": ["action_restart"]}]}]
    }
    with pytest.raises(ValidationError):
        MarkerConfig._validate_config(sample_json)


def test_config_missing_required_condition_label():
    sample_json = {"markers": [{"marker": "no_restart"}]}
    with pytest.raises(ValidationError):
        MarkerConfig._validate_config(sample_json)


def test_config_invalid_operator():
    sample_json = {
        "markers": [
            {
                "marker": "no_restart",
                "operator": "XOR",
                "condition": [{"action_not_executed": ["action_restart"]}],
            },
        ]
    }
    with pytest.raises(ValidationError):
        MarkerConfig._validate_config(sample_json)


def test_config_invalid_event():
    sample_json = {
        "markers": [
            {
                "marker": "no_restart",
                "operator": "XOR",
                "condition": [{"action": ["action_restart"]}],
            },
        ]
    }
    with pytest.raises(ValidationError):
        MarkerConfig._validate_config(sample_json)
