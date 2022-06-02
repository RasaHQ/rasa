import csv
from pathlib import Path
from typing import Dict, List, Text, Tuple
import itertools

import pytest
import numpy as np

from rasa.core.evaluation.marker_stats import (
    EventMetaData,
    MarkerStatistics,
    compute_statistics,
)


def test_compute_statistics_on_empty_set():
    result = compute_statistics([])
    assert result["count"] == 0
    assert all(np.isnan(value) for key, value in result.items() if key != "count")
    # we see the same keys as in those cases where we can evaluate something
    dummy_result_non_empty = compute_statistics([1, 2])
    assert set(dummy_result_non_empty.keys()) == set(result.keys())


def test_compute_statistics_simple_check():
    stats = compute_statistics([1, 2, 9, 0])
    assert stats["count"] == 4
    assert stats["min"] == 0
    assert stats["max"] == 9
    assert stats["mean"] == 3
    assert stats["median"] == 1.5  # this is no bug, it is a convention numpy follows


def _generate_random_example_for_one_session_and_one_marker(
    rng: np.random.Generator,
) -> Tuple[List[EventMetaData], List[int]]:
    """Generates a random marker extraction result for a single session and marker.

    Args:
        rng: a random number generator
    Returns:
        the event list representing the marker extraction result as well as
        the plain list of numbers used as "preceding user turns" in that extraction
        result
    """
    applied = int(rng.choice(10))
    all_preceding_user_turn_numbers = [int(rng.choice(20)) for _ in range(applied)]
    event_list = [
        EventMetaData(
            idx=int(rng.choice(100)), preceding_user_turns=preceding_user_turns
        )
        for preceding_user_turns in all_preceding_user_turn_numbers
    ]
    return event_list, all_preceding_user_turn_numbers


# For every evaluated session, we obtain as a result a mapping of the evaluated
# marker names to the meta data for the respective relevant events.
PerMarkerResults = Dict[Text, List[EventMetaData]]
# We collect the "number of preceding user turns" that we generate randomly and
# compare them to the actual results later. In the following, the inner `List[int]`
# contains the "number of preceding user turns" created for one session (and the
# respective dialogue):
PerMarkerCollectedNumbers = Dict[Text, List[List[int]]]


def _generate_random_examples(
    rng: np.random.Generator,
    num_markers: int = 3,
    num_sessions_min: int = 2,
    num_sessions_max: int = 10,
) -> Tuple[List[PerMarkerResults], PerMarkerCollectedNumbers]:
    """Generates a random number of random marker extraction results for some markers.

    Args:
        rng: a random number generator
        num_markers: the number of markers to be imitated
    Returns:
        a list containing a dictionary of the marker extraction results per marker,
        as well as a collection of the plain list of numbers used as "preceding user
        turns" in that extraction results
    """
    num_sessions = int(rng.integers(low=num_sessions_min, high=num_sessions_max + 1))
    markers = [f"marker{idx}" for idx in range(num_markers)]
    per_session_results: List[PerMarkerResults] = []
    preceding_user_turn_numbers_used_per_marker: PerMarkerCollectedNumbers = {
        marker: [] for marker in markers
    }
    for _ in range(num_sessions - 1):  # we append one more later
        result_dict = {}
        for marker in markers:
            (
                event_list,
                num_list,
            ) = _generate_random_example_for_one_session_and_one_marker(rng=rng)
            result_dict[marker] = event_list
            preceding_user_turn_numbers_used_per_marker[marker].append(num_list)
        per_session_results.append(result_dict)
    # append a session where we didn't find any marker
    per_session_results.append({marker: [] for marker in markers})
    for marker in preceding_user_turn_numbers_used_per_marker:
        preceding_user_turn_numbers_used_per_marker[marker].append([])
    return per_session_results, preceding_user_turn_numbers_used_per_marker


@pytest.mark.parametrize("seed", [2345, 5654, 2345234])
def test_process_results_per_session(seed: int):
    rng = np.random.default_rng(seed=seed)

    (
        per_session_results,
        preceding_user_turn_numbers_used_per_marker,
    ) = _generate_random_examples(num_markers=3, rng=rng)
    markers = sorted(preceding_user_turn_numbers_used_per_marker.keys())
    num_sessions = len(per_session_results)

    stats = MarkerStatistics()
    sender_ids = []
    session_indices = []
    for session_idx, results in enumerate(per_session_results):
        sender_id = str(rng.choice(100))
        session_idx = int(rng.choice(100))
        stats.process(
            session_idx=session_idx,
            sender_id=sender_id,
            meta_data_on_relevant_events_per_marker=results,
        )
        sender_ids.append(sender_id)
        session_indices.append(session_idx)

    assert stats.num_sessions == len(per_session_results)
    for marker in markers:
        for idx in range(num_sessions):
            expected_stats = compute_statistics(
                preceding_user_turn_numbers_used_per_marker[marker][idx]
            )
            for stat_name, stat_value in expected_stats.items():
                assert (
                    pytest.approx(
                        stats.session_results[marker][stat_name][idx], nan_ok=True
                    )
                    == stat_value
                )
    for idx in range(num_sessions):
        assert stats.session_identifier[idx] == (sender_ids[idx], session_indices[idx])


@pytest.mark.parametrize("seed", [2345, 5654, 2345234])
def test_process_results_overall(seed: int):
    rng = np.random.default_rng(seed=seed)
    (
        per_session_results,
        preceding_user_turn_numbers_used_per_marker,
    ) = _generate_random_examples(num_markers=3, rng=rng)
    markers = sorted(preceding_user_turn_numbers_used_per_marker.keys())
    num_sessions = len(per_session_results)

    stats = MarkerStatistics()
    for session_idx, results in enumerate(per_session_results):
        stats.process(
            session_idx=session_idx,
            sender_id=str(rng.choice(100)),
            meta_data_on_relevant_events_per_marker=results,
        )

    assert stats.num_sessions == num_sessions
    for marker in markers:
        # count how often we generated some results for a session:
        number_lists = preceding_user_turn_numbers_used_per_marker[marker]
        applied_at_least_once = sum(len(sub_list) > 0 for sub_list in number_lists)
        # and compare that to the expected count:
        assert stats.count_if_applied_at_least_once[marker] == applied_at_least_once
        # check if we collected the all the "preceding user turn numbers"
        concatenated_numbers = list(
            itertools.chain.from_iterable(
                preceding_user_turn_numbers_used_per_marker[marker]
            )
        )
        assert stats.num_preceding_user_turns_collected[marker] == concatenated_numbers


@pytest.mark.parametrize("seed", [2345, 5654, 2345234])
def test_overall_statistics_to_csv(tmp_path: Path, seed: int):
    rng = np.random.default_rng(seed=seed)
    (
        per_session_results,
        preceding_user_turn_numbers_used_per_marker,
    ) = _generate_random_examples(
        num_markers=3, rng=rng, num_sessions_min=10, num_sessions_max=20
    )
    markers = sorted(preceding_user_turn_numbers_used_per_marker.keys())
    num_sessions = len(per_session_results)

    stats = MarkerStatistics()
    for session_idx, results in enumerate(per_session_results):
        stats.process(
            session_idx=session_idx,
            sender_id=str(rng.choice(100)),
            meta_data_on_relevant_events_per_marker=results,
        )

    tmp_file = tmp_path / "test.csv"
    stats.overall_statistic_to_csv(path=tmp_file)

    with tmp_file.open(mode="r") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    assert rows[0] == {
        "sender_id": "all",
        "session_idx": "nan",
        "marker": "-",
        "statistic": "total_number_of_sessions",
        "value": str(num_sessions),
    }

    num_digits = 3
    row_idx = 1
    for marker_name in markers:
        assert rows[row_idx] == {
            "sender_id": "all",
            "session_idx": "nan",
            "marker": marker_name,
            "statistic": "number_of_sessions_where_marker_applied_at_least_once",
            "value": str(stats.count_if_applied_at_least_once[marker_name]),
        }
        row_idx += 1
        assert rows[row_idx] == {
            "sender_id": "all",
            "session_idx": "nan",
            "marker": marker_name,
            "statistic": "percentage_of_sessions_where_marker_applied_at_least_once",
            "value": str(
                round(
                    stats.count_if_applied_at_least_once[marker_name]
                    / num_sessions
                    * 100,
                    num_digits,
                )
            ),
        }
        row_idx += 1

    for marker_name in markers:
        statistics = compute_statistics(
            stats.num_preceding_user_turns_collected[marker_name]
        )
        for stat_name, stat_value in statistics.items():
            assert rows[row_idx] == {
                "sender_id": "all",
                "session_idx": "nan",
                "marker": marker_name,
                "statistic": MarkerStatistics._add_num_user_turns_str_to(stat_name),
                "value": str(round(stat_value, num_digits)),
            }
            row_idx += 1


@pytest.mark.parametrize("seed", [2345, 5654, 2345234])
def test_per_session_statistics_to_csv(tmp_path: Path, seed: int):

    rng = np.random.default_rng(seed=seed)
    (
        per_session_results,
        preceding_user_turn_numbers_used_per_marker,
    ) = _generate_random_examples(
        num_markers=3, rng=rng, num_sessions_min=10, num_sessions_max=20
    )
    markers = sorted(preceding_user_turn_numbers_used_per_marker.keys())

    stats = MarkerStatistics()
    for session_idx, results in enumerate(per_session_results):
        stats.process(
            session_idx=session_idx,
            sender_id=str(rng.choice(100)),
            meta_data_on_relevant_events_per_marker=results,
        )

    tmp_file = tmp_path / "test.csv"
    stats.per_session_statistics_to_csv(path=tmp_file)

    with tmp_file.open(mode="r") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    actual_information = {
        (row["sender_id"], row["session_idx"], row["marker"], row["statistic"]): row[
            "value"
        ]
        for row in rows
    }
    num_digits = 3
    expected_information = {
        (
            sender_id,
            str(session_idx),
            marker_name,
            MarkerStatistics._add_num_user_turns_str_to(stat_name),
        ): str(value)
        if np.isnan(value)
        else str(round(value, num_digits))
        for marker_name in markers
        for stat_name, values in stats.session_results[marker_name].items()
        for (sender_id, session_idx), value in zip(stats.session_identifier, values)
    }

    assert actual_information == expected_information
