import os
import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict
from typing import Dict, Text, Union, Any, List, Tuple
from pathlib import Path
import json
import numpy as np


class MarkerStats(TypedDict):
    """A TypedDict for statistics computed over extracted markers."""

    n: int
    mean: float
    median: float
    min: int
    max: int


def load_extracted_markers_json_file(path: Union[Text, Path]) -> List:
    """Reads a json marker file.

    Args:
        path: path to a json file.
    """
    path = os.path.abspath(path)
    with open(path) as json_file:
        extracted_markers = json.load(json_file)
        return extracted_markers


def compute_summary_stats(data_points: Union[List, np.ndarray]) -> MarkerStats:
    """Computes summary statistics for a given array.

    Computes size, mean, median, min, and max.
    If size is == 0 returns np.nan for mean, median.

    Args:
        data_points: can be a numpy array or a list of numbers.
    """
    if np.size(data_points) > 0:
        stats: MarkerStats = {
            "n": int(np.size(data_points)),
            "mean": float(np.mean(data_points)),
            "median": float(np.median(data_points)),
            "min": int(np.min(data_points)),
            "max": int(np.max(data_points)),
        }
        return stats
    else:
        empty_stats: MarkerStats = {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
        return empty_stats


def compute_single_tracker_stats(
    single_tracker_markers: Dict[str, Any]
) -> Dict[str, MarkerStats]:
    """Computes summary statistics for a single tracker."""
    tracker_stats = {}
    for marker in single_tracker_markers["markers"]:
        tracker_stats[marker["marker"]] = compute_summary_stats(
            marker["num_preceding_user_turns"]
        )
    return tracker_stats


def compute_multi_tracker_stats(
    multi_tracker_markers: list,
) -> Tuple[Dict, Dict[Any, Dict[str, MarkerStats]]]:
    """Computes summary statistics for multiple trackers."""
    overall_stats = {"num_trackers": len(multi_tracker_markers)}
    per_tracker_stats = {}
    per_marker_values = {}

    for tracker in multi_tracker_markers:
        per_tracker_stats[tracker["tracker_ID"]] = compute_single_tracker_stats(tracker)
        for marker in tracker["markers"]:
            # append raw values
            per_marker_values.setdefault(marker["marker"], []).extend(
                marker["num_preceding_user_turns"]
            )

    for marker_name in per_marker_values.keys():
        # compute overall statistics
        overall_stats[marker_name] = compute_summary_stats(
            per_marker_values[marker_name]
        )

    return overall_stats, per_tracker_stats


def write_stats(path: Union[Text, Path], stats: dict, per_tracker_stats: dict) -> None:
    """Outputs statistics to JSON file."""
    path = os.path.abspath(path)
    data = {"marker_stats": stats, "tracker_stats": per_tracker_stats}
    with open(path, "w") as outfile:
        json_str = json.dumps(data, default=np_encoder, indent=2)
        outfile.write(json_str)


def np_encoder(obj: Any) -> Any:
    """Encodes numpy array values to make them JSON serializable.

    adapted from: https://bit.ly/3ajjTwp
    """
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
