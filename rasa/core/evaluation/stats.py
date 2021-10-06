import json
import numpy as np


def read_json_marker_file(path) -> list:
    with open(path) as json_file:
        marker_file = json.load(json_file)
        return marker_file


def get_summary_stats(data_points) -> dict:
    """Computes summary statistics.

    computes size, mean, median, min, and max.
    if size is == 0 returns np.nan for mean, median.
    """
    summary_stats = dict()
    summary_stats['n'] = np.size(data_points)

    if np.size(data_points) > 0:
        summary_stats['mean'] = np.mean(data_points)
        summary_stats['median'] = np.median(data_points)
        summary_stats['min'] = np.min(data_points)
        summary_stats['max'] = np.max(data_points)
    else:
        summary_stats['mean'] = np.nan
        summary_stats['median'] = np.nan
        summary_stats['min'] = np.nan
        summary_stats['max'] = np.nan

    return summary_stats


def get_per_tracker_stats(single_tracker_markers) -> dict:
    tracker_stats = dict()
    for item in single_tracker_markers['markers']:
        data_points = item['num_preceding_user_turns']
        tracker_stats[item['marker']] = get_summary_stats(data_points)
    return tracker_stats

