from rasa.core.evaluation.markers import MarkerConfig, InvalidMarkersConfig
import pytest
from typing import Text


@pytest.fixture
def simple_marker_config_json() -> dict:
    """Returns a json dict equivalent to simple_marker_config fixture"""
    sample_json = {
        "markers": [
            {
                "marker": "carbon_offset_calculated",
                "conditions": [
                    {"type": "AND"},
                    {
                        "slot_filled": [
                            "travel_flight_class",
                            "travel_departure",
                            "travel_destination",
                        ]
                    },
                    {"action_executed": ["provide_carbon_estimate"]},
                ],
            }
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
                        "slot_filled": [
                            "travel_flight_class",
                            "travel_departure",
                            "travel_destination",
                        ]
                    }
                ],
            },
            {
                "marker": "carbon_offset_calculated",
                "type": "AND",
                "condition": [
                    {
                        "slot_filled": [
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
      conditions:
      - type: AND
      - slot_filled:
        - travel_flight_class
        - travel_departure
        - travel_destination
      - action_executed:
        - provide_carbon_estimate
    """
    yaml_as_dict = MarkerConfig.from_yaml(simple_yaml_markers_config)
    assert yaml_as_dict == simple_marker_config_json


def test_load_invalid_path():
    """Checks that the correct exception is raised when an invalid path is supplied"""
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
