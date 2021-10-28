from pathlib import Path
from rasa.shared.core.domain import Domain
from rasa.core.tracker_store import SQLTrackerStore
from typing import Callable, Text, Tuple, Dict, Any
import csv

import pytest
from _pytest.pytester import RunResult

import rasa.cli.evaluate

from rasa.shared.core.events import ActionExecuted, SlotSet, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.constants import ACTION_SESSION_START_NAME
from tests.conftest import write_endpoint_config_to_yaml


@pytest.fixture
def marker_sqlite_tracker(tmp_path: Path) -> Tuple[SQLTrackerStore, Text]:
    domain = Domain.empty()
    db_path = str(tmp_path / "rasa.db")
    tracker_store = SQLTrackerStore(dialect="sqlite", db=db_path)
    for i in range(5):
        tracker = DialogueStateTracker(str(i), None)
        tracker.update_with_events([SlotSet(str(j), "slot") for j in range(5)], domain)
        tracker.update(ActionExecuted(ACTION_SESSION_START_NAME))
        tracker.update(UserUttered("hello"))
        tracker.update_with_events(
            [SlotSet(str(5 + j), "slot") for j in range(5)], domain
        )
        tracker_store.save(tracker)

    return tracker_store, db_path


def write_markers_config_to_yaml(
    path: Path, data: Dict[Text, Any], markers_filename: Text = "markers.yml"
) -> Path:
    markers_path = path / markers_filename

    # write markers config to file
    rasa.shared.utils.io.write_yaml(data, markers_path)
    return markers_path


def test_evaluate_markers_help(run: Callable[..., RunResult]):
    output = run("evaluate", "markers", "--help")

    help_text = """usage: rasa evaluate markers [-h] [-v] [-vv] [--quiet] [--config CONFIG]
                             [--no-stats | --stats-file STATS_FILE]
                             [--endpoints ENDPOINTS] [-d DOMAIN]
                             output_filename {first_n,sample,all} ..."""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_evaluate_markers_first_n_help(run: Callable[..., RunResult]):
    # We need to specify an output_filename as that's the first positional parameter
    output = run("evaluate", "markers", "--no-stats", "test.csv", "first_n", "--help")

    help_text = """usage: rasa evaluate markers output_filename first_n [-h] [-v] [-vv] [--quiet]
                                                     count"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_evaluate_markers_sample_help(run: Callable[..., RunResult]):
    # We need to specify an output_filename as that's the first positional parameter
    output = run("evaluate", "markers", "test.csv", "sample", "--help")

    help_text = """usage: rasa evaluate markers output_filename sample [-h] [-v] [-vv] [--quiet]
                                                    [--seed SEED]
                                                    count"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_evaluate_markers_all_help(run: Callable[..., RunResult]):
    # We need to specify an output_filename as that's the first positional parameter
    output = run("evaluate", "markers", "test.csv", "all", "--help")

    help_text = (
        """usage: rasa evaluate markers output_filename all [-h] [-v] [-vv] [--quiet]"""
    )

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_markers_cli_results_save_correctly(
    marker_sqlite_tracker: Tuple[SQLTrackerStore, Text], tmp_path: Path
):
    _, db_path = marker_sqlite_tracker

    endpoints_path = write_endpoint_config_to_yaml(
        tmp_path,
        {"tracker_store": {"type": "sql", "db": db_path.replace("\\", "\\\\")},},
    )

    markers_path = write_markers_config_to_yaml(
        tmp_path, {"marker1": {"slot_is_set": "2"}, "marker2": {"slot_is_set": "7"}}
    )

    results_path = tmp_path / "results.csv"

    rasa.cli.evaluate._run_markers(
        None, 10, endpoints_path, "first_n", markers_path, results_path, None
    )

    with open(results_path, "r") as results:
        result_reader = csv.DictReader(results)
        senders = set()

        for row in result_reader:
            senders.add(row["sender_id"])
            if row["marker_name"] == "marker1":
                assert row["dialogue_id"] == "0"
                assert int(row["event_id"]) >= 2
                assert row["num_preceding_user_turns"] == "0"

            if row["marker_name"] == "marker2":
                assert row["dialogue_id"] == "1"
                assert int(row["event_id"]) >= 3
                assert row["num_preceding_user_turns"] == "1"

        assert len(senders) == 5
