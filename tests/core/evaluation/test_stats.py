import os
import math
import numpy as np
import json
from rasa.core.evaluation.stats import (
    load_extracted_markers_json_file,
    compute_multi_tracker_stats,
    compute_summary_stats,
    compute_single_tracker_stats,
    write_stats,
)


def test_compute_summary_stats():
    """Tests compute summary stats."""
    a = [20, 30, 32]
    stats_a = compute_summary_stats(a)
    assert stats_a["n"] == 3
    assert math.isclose(stats_a["mean"], 27.33, abs_tol=0.01)
    assert stats_a["median"] == 30
    assert stats_a["min"] == 20
    assert stats_a["max"] == 32

    b = [15]
    stats_b = compute_summary_stats(b)
    assert stats_b["n"] == 1
    assert stats_b["mean"] == 15
    assert stats_b["median"] == 15
    assert stats_b["min"] == 15
    assert stats_b["max"] == 15


def test_compute_summary_stats_empty():
    """Tests compute summary stats on an empty array."""
    stats = compute_summary_stats([])
    assert stats == {
        "n": 0,
        "mean": np.nan,
        "median": np.nan,
        "min": np.nan,
        "max": np.nan,
    }


def test_compute_single_tracker_stats(extracted_markers_json):
    """Tests compute summary stats for a single tracker."""
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
    extracted_markers = load_extracted_markers_json_file(extracted_markers_json)
    stats = compute_single_tracker_stats(extracted_markers[0])
    assert stats == expected_output


def test_compute_single_tracker_stats_some_empty(extracted_markers_json):
    """Tests compute summary stats for a single tracker with empty values."""
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
    extracted_markers = load_extracted_markers_json_file(extracted_markers_json)
    stats = compute_single_tracker_stats(extracted_markers[1])
    assert stats == expected_output


def test_compute_multi_tracker_stats(extracted_markers_json):
    """Tests compute summary stats on multiple trackers."""
    expected_overall_stats = {
        "num_trackers": 3,
        "no_restart": {"n": 2, "mean": 16.5, "median": 16.5, "min": 13, "max": 20},
        "all_required_data_gathered": {
            "n": 4,
            "mean": 11.0,
            "median": 8.0,
            "min": 8,
            "max": 20,
        },
        "carbon_offset_calculated": {
            "n": 6,
            "mean": 12.0,
            "median": 11.0,
            "min": 8,
            "max": 22,
        },
        "chitchat_and_FAQ": {"n": 4, "mean": 6.5, "median": 5.5, "min": 2, "max": 13},
        "dialog_expected_order": {
            "n": 4,
            "mean": 10.0,
            "median": 8.0,
            "min": 5,
            "max": 19,
        },
    }
    expected_per_tracker_stats = {
        "8b32209b-2015-4c00-97e2-2757836cf8e9": {
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
            "chitchat_and_FAQ": {
                "n": 3,
                "mean": 8.0,
                "median": 9.0,
                "min": 2,
                "max": 13,
            },
            "dialog_expected_order": {
                "n": 1,
                "mean": 5.0,
                "median": 5.0,
                "min": 5,
                "max": 5,
            },
        },
        "8b32209b-2015-4c00-97e2-275546548e9": {
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
        },
        "8b32209b-2015-4c00-97e2-2756588e9": {
            "no_restart": {
                "n": 0,
                "mean": np.nan,
                "median": np.nan,
                "min": np.nan,
                "max": np.nan,
            },
            "all_required_data_gathered": {
                "n": 2,
                "mean": 14.0,
                "median": 14.0,
                "min": 8,
                "max": 20,
            },
            "carbon_offset_calculated": {
                "n": 2,
                "mean": 16.0,
                "median": 16.0,
                "min": 10,
                "max": 22,
            },
            "chitchat_and_FAQ": {
                "n": 1,
                "mean": 2.0,
                "median": 2.0,
                "min": 2,
                "max": 2,
            },
            "dialog_expected_order": {
                "n": 2,
                "mean": 14.0,
                "median": 14.0,
                "min": 9,
                "max": 19,
            },
        },
    }
    extracted_markers = load_extracted_markers_json_file(extracted_markers_json)
    stats, per_tracker_stats = compute_multi_tracker_stats(extracted_markers)
    assert stats == expected_overall_stats
    assert per_tracker_stats == expected_per_tracker_stats


def test_write_stats(extracted_markers_json, marker_stats_output_json):
    """Tests write summary stats."""
    extracted_markers = load_extracted_markers_json_file(extracted_markers_json)
    stats, per_tracker_stats = compute_multi_tracker_stats(extracted_markers)
    write_stats(marker_stats_output_json, stats, per_tracker_stats)
    assert os.path.exists(marker_stats_output_json)
    os.remove(marker_stats_output_json)
