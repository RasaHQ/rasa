from pathlib import Path
from typing import Callable, Text, Tuple, Dict, Any
import csv

import pytest
from _pytest.pytester import RunResult

import rasa.cli.evaluate

from rasa.shared.core.events import ActionExecuted, SlotSet, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.constants import ACTION_SESSION_START_NAME
from rasa.shared.core.domain import Domain
from rasa.core.tracker_store import SQLTrackerStore
from rasa.cli.evaluate import STATS_SESSION_SUFFIX, STATS_OVERALL_SUFFIX
from tests.conftest import write_endpoint_config_to_yaml

from tests.cli.conftest import RASA_EXE


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

    help_text = f"""usage: {RASA_EXE} evaluate markers [-h] [-v] [-vv] [--quiet]
    [--logging-config-file LOGGING_CONFIG_FILE]
    {{first_n,sample_n,all}} ..."""

    lines = [line.strip() for line in help_text.split("\n")]
    # expected help text lines should appear somewhere in the output
    printed_help = set([line.strip() for line in output.outlines])
    for line in lines:
        assert line in printed_help


def test_evaluate_markers_first_n_help(run: Callable[..., RunResult]):
    # We need to specify an output_filename as that's the first positional parameter
    output = run("evaluate", "markers", "first_n", "--help")

    help_text = f"""usage: {RASA_EXE} evaluate markers first_n [-h] [-v] [-vv] [--quiet]
    [--logging-config-file LOGGING_CONFIG_FILE]
    [--config CONFIG]
    [--no-stats | --stats-file-prefix [STATS_FILE_PREFIX]]
    [--endpoints ENDPOINTS] [-d DOMAIN]
    count output_filename"""

    lines = [line.strip() for line in help_text.split("\n")]
    # expected help text lines should appear somewhere in the output
    printed_help = set([line.strip() for line in output.outlines])
    for line in lines:
        assert line in printed_help


def test_evaluate_markers_sample_n_help(run: Callable[..., RunResult]):
    # We need to specify an output_filename as that's the first positional parameter
    output = run("evaluate", "markers", "sample_n", "--help")

    help_text = f"""usage: {RASA_EXE} evaluate markers sample_n [-h] [-v] [-vv] [--quiet]
    [--logging-config-file LOGGING_CONFIG_FILE]
    [--seed SEED] [--config CONFIG]
    [--no-stats | --stats-file-prefix [STATS_FILE_PREFIX]]
    [--endpoints ENDPOINTS] [-d DOMAIN]
    count output_filename"""  # noqa: E501

    lines = [line.strip() for line in help_text.split("\n")]
    # expected help text lines should appear somewhere in the output
    printed_help = set([line.strip() for line in output.outlines])
    for line in lines:
        assert line in printed_help


def test_evaluate_markers_all_help(run: Callable[..., RunResult]):
    # We need to specify an output_filename as that's the first positional parameter
    output = run("evaluate", "markers", "all", "--help")

    help_text = f"""usage: {RASA_EXE} evaluate markers all [-h] [-v] [-vv] [--quiet]
    [--logging-config-file LOGGING_CONFIG_FILE]
    [--config CONFIG]
    [--no-stats | --stats-file-prefix [STATS_FILE_PREFIX]]
    [--endpoints ENDPOINTS] [-d DOMAIN]
    output_filename"""

    lines = [line.strip() for line in help_text.split("\n")]
    # expected help text lines should appear somewhere in the output
    printed_help = set([line.strip() for line in output.outlines])
    for line in lines:
        assert line in printed_help


def test_markers_cli_results_save_correctly(
    marker_sqlite_tracker: Tuple[SQLTrackerStore, Text], tmp_path: Path
):
    _, db_path = marker_sqlite_tracker

    endpoints_path = write_endpoint_config_to_yaml(
        tmp_path,
        {"tracker_store": {"type": "sql", "db": db_path.replace("\\", "\\\\")}},
    )

    markers_path = write_markers_config_to_yaml(
        tmp_path, {"marker1": {"slot_was_set": "2"}, "marker2": {"slot_was_set": "7"}}
    )

    results_path = tmp_path / "results.csv"
    stats_file_prefix = tmp_path / "statistics"

    rasa.cli.evaluate._run_markers(
        seed=None,
        count=10,
        endpoint_config=endpoints_path,
        strategy="first_n",
        domain_path=None,
        config=markers_path,
        output_filename=results_path,
        stats_file_prefix=stats_file_prefix,
    )

    for expected_output in [
        results_path,
        tmp_path / ("statistics" + STATS_SESSION_SUFFIX),
        tmp_path / ("statistics" + STATS_OVERALL_SUFFIX),
    ]:
        with expected_output.open(mode="r") as results:
            result_reader = csv.DictReader(results)
            # Loop over entire file to ensure nothing in the file causes any errors
            for _ in result_reader:
                continue
