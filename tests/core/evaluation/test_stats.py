import pytest
from rasa.core.evaluation.stats import (
    read_json_marker_file,
    get_summary_stats,
    get_per_tracker_stats,
)
import math
import numpy as np


@pytest.fixture
def simple_marker_file_json() -> dict:
    """Returns a json markers file"""
    sample_json = {
        "tracker_ID": "8b32209b-2015-4c00-97e2-2757836cf8e9",
        "bot-version": "<version-number>",
        "markers": [
            {
                "marker": "no_restart",
                "num_preceding_user_turns": [20],
                "timestamps": [63123.21442],
            },
            {
                "marker": "all_required_data_gathered",
                "num_preceding_user_turns": [8],
                "timestamps": [24123.21452],
            },
            {
                "marker": "carbon_offset_calculated",
                "num_preceding_user_turns": [8, 12],
                "timestamps": [24723.21485, 33100.24350],
            },
            {
                "marker": "chitchat_and_FAQ",
                "num_preceding_user_turns": [2, 9, 13],
                "timestamps": [14123.21452, 25163.21452, 35173.21452],
            },
            {
                "marker": "dialog_expected_order",
                "num_preceding_user_turns": [5],
                "timestamps": [20123.54555],
            },
        ],
    }
    return sample_json


def test_get_summary_stats():
    a = [20, 30, 32]
    stats_a = get_summary_stats(a)
    assert stats_a["n"] == 3
    assert math.isclose(stats_a["mean"], 27.33, abs_tol=0.01)
    assert stats_a["median"] == 30
    assert stats_a["min"] == 20
    assert stats_a["max"] == 32

    b = [15]
    stats_b = get_summary_stats(b)
    assert stats_b["n"] == 1
    assert stats_b["mean"] == 15
    assert stats_b["median"] == 15
    assert stats_b["min"] == 15
    assert stats_b["max"] == 15


def test_get_summary_stats_raises_error():
    stats = get_summary_stats([])
    assert stats == {
        "n": 0,
        "mean": np.nan,
        "median": np.nan,
        "min": np.nan,
        "max": np.nan,
    }


def test_get_per_tracker_stats(marker_output_json):
    expected_output = {
        "no_restart": {"n": 1, "mean": 20.0, "median": 20.0, "min": 20, "max": 20},
        "all_required_data_gathered": {
            "n": 1,
            "mean": 8.0,
            "median": 8.0,
            "min": 8,
            "max": 8,
        },
        "carbon_offset_calculated": {
            "n": 2,
            "mean": 10.0,
            "median": 10.0,
            "min": 8,
            "max": 12,
        },
        "chitchat_and_FAQ": {"n": 3, "mean": 8.0, "median": 9.0, "min": 2, "max": 13},
        "dialog_expected_order": {
            "n": 1,
            "mean": 5.0,
            "median": 5.0,
            "min": 5,
            "max": 5,
        },
    }
    marker_results = read_json_marker_file(marker_output_json)
    stats = get_per_tracker_stats(marker_results[0])
    assert stats == expected_output


def test_get_per_tracker_stats_some_empty_markers(marker_output_json):
    expected_output = {
        "no_restart": {"n": 1, "mean": 13.0, "median": 13.0, "min": 13, "max": 13},
        "all_required_data_gathered": {
            "n": 1,
            "mean": 8.0,
            "median": 8.0,
            "min": 8,
            "max": 8,
        },
        "carbon_offset_calculated": {
            "n": 2,
            "mean": 10.0,
            "median": 10.0,
            "min": 8,
            "max": 12,
        },
        "chitchat_and_FAQ": {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        },
        "dialog_expected_order": {
            "n": 1,
            "mean": 7.0,
            "median": 7.0,
            "min": 7,
            "max": 7,
        },
    }
    marker_results = read_json_marker_file(marker_output_json)
    stats = get_per_tracker_stats(marker_results[1])
    assert stats == expected_output
