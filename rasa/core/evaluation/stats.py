import os
from pathlib import Path
from typing import Dict, Text, Union, Any
import json
import numpy as np


def read_json_marker_file(path: Union[Text, Path]) -> list:
    """Reads a json marker file."""
    path = os.path.abspath(path)
    with open(path) as json_file:
        marker_file = json.load(json_file)
        return marker_file


def get_summary_stats(data_points: Union[list, np.ndarray]) -> Dict[str, float]:
    """Computes summary statistics on a given array.

    computes size, mean, median, min, and max.
    if size is == 0 returns np.nan for mean, median.
    """
    summary_stats = dict()
    summary_stats["n"] = np.size(data_points)

    if np.size(data_points) > 0:
        summary_stats["mean"] = np.mean(data_points)
        summary_stats["median"] = np.median(data_points)
        summary_stats["min"] = np.min(data_points)
        summary_stats["max"] = np.max(data_points)
    else:
        summary_stats["mean"] = np.nan
        summary_stats["median"] = np.nan
        summary_stats["min"] = np.nan
        summary_stats["max"] = np.nan

    return summary_stats


def get_per_tracker_stats(single_tracker_markers: Dict[str, Any]) -> dict:
    tracker_stats = dict()
    for item in single_tracker_markers["markers"]:
        tracker_stats[item["marker"]] = get_summary_stats(
            item["num_preceding_user_turns"]
        )
    return tracker_stats


def compute_stats(per_tracker_marker: list) -> (dict, dict):
    stats = dict()
    stats["num_trackers"] = len(per_tracker_marker)

    per_marker_values = dict()
    per_tracker_stats = dict()
    for tracker in per_tracker_marker:

        per_tracker_stats[tracker["tracker_ID"]] = get_per_tracker_stats(tracker)
        for marker in tracker["markers"]:
            per_marker_values.setdefault(marker["marker"], []).extend(
                marker["num_preceding_user_turns"]
            )

    for marker_name in per_marker_values.keys():
        stats[marker_name] = get_summary_stats(per_marker_values[marker_name])

    return stats, per_tracker_stats


def write_statistics(
    path: Union[Text, Path], stats: dict, per_tracker_stats: dict
) -> None:
    path = os.path.abspath(path)

    data = {
        'marker_stats': stats,
        'tracker_stats': per_tracker_stats
    }
    with open(path, "w") as outfile:
        json_str = json.dumps(data, default=np_encoder, indent=2)
        outfile.write(json_str)


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

