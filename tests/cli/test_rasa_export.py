import argparse
from pathlib import Path
from typing import Callable, Optional, Dict, Text, List, Tuple, Any
from unittest.mock import Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import RunResult
from ruamel.yaml.scalarstring import SingleQuotedScalarString

import rasa.core.utils as rasa_core_utils
from rasa.cli import export
from rasa.core.brokers.pika import PikaEventBroker
from rasa.core.events import UserUttered
from rasa.core.trackers import DialogueStateTracker
from rasa.exceptions import PublishingError, NoEventsToMigrateError
from tests.conftest import (
    MockExporter,
    random_user_uttered_event,
    write_endpoint_config_to_yaml,
)


def test_export_help(run: Callable[..., RunResult]):
    output = run("export", "--help")

    help_text = """usage: rasa export [-h] [-v] [-vv] [--quiet] [--endpoints ENDPOINTS]
                   [--minimum-timestamp MINIMUM_TIMESTAMP]
                   [--maximum-timestamp MAXIMUM_TIMESTAMP]
                   [--conversation-ids CONVERSATION_IDS]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


@pytest.mark.parametrize(
    "minimum_timestamp,maximum_timestamp",
    [(2, 3), (None, 5.5), (None, None), (5, None)],
)
def test_validate_timestamp_options(
    minimum_timestamp: Optional[float], maximum_timestamp: Optional[float]
):
    args = argparse.Namespace()
    args.minimum_timestamp = (
        str(minimum_timestamp) if minimum_timestamp is not None else None
    )
    args.maximum_timestamp = (
        str(maximum_timestamp) if maximum_timestamp is not None else None
    )

    # no error is raised
    # noinspection PyProtectedMember
    export._assert_max_timestamp_is_greater_than_min_timestamp(args)


def test_validate_timestamp_options_with_invalid_timestamps():
    args = argparse.Namespace(minimum_timestamp=3, maximum_timestamp=2)
    with pytest.raises(SystemExit):
        # noinspection PyProtectedMember
        export._assert_max_timestamp_is_greater_than_min_timestamp(args)


# noinspection PyProtectedMember
def test_get_event_broker_and_tracker_store_from_endpoint_config(tmp_path: Path):
    # write valid config to file
    endpoints_path = write_endpoint_config_to_yaml(
        tmp_path,
        {
            "event_broker": {
                "type": "sql",
                "db": str(tmp_path / "rasa.db").replace("\\", "\\\\"),
            },
            "tracker_store": {"type": "sql"},
        },
    )

    available_endpoints = rasa_core_utils.read_endpoints_from_path(endpoints_path)

    # fetching the event broker is successful
    assert export._get_event_broker(available_endpoints)
    assert export._get_tracker_store(available_endpoints)


# noinspection PyProtectedMember
def test_get_event_broker_from_endpoint_config_error_exit(tmp_path: Path):
    # write config without event broker to file
    endpoints_path = write_endpoint_config_to_yaml(
        tmp_path, {"tracker_store": {"type": "sql"}}
    )

    available_endpoints = rasa_core_utils.read_endpoints_from_path(endpoints_path)

    with pytest.raises(SystemExit):
        assert export._get_event_broker(available_endpoints)


def test_get_tracker_store_from_endpoint_config_error_exit(tmp_path: Path):
    # write config without event broker to file
    endpoints_path = write_endpoint_config_to_yaml(tmp_path, {})

    available_endpoints = rasa_core_utils.read_endpoints_from_path(endpoints_path)

    with pytest.raises(SystemExit):
        # noinspection PyProtectedMember
        assert export._get_tracker_store(available_endpoints)


@pytest.mark.parametrize(
    "requested_ids,expected",
    [("id1", ["id1"]), ("id1,id2", ["id1", "id2"]), (None, None), ("", None)],
)
def test_get_requested_conversation_ids(
    requested_ids: Optional[Text], expected: Optional[List[Text]]
):
    # noinspection PyProtectedMember
    assert export._get_requested_conversation_ids(requested_ids) == expected


def test_prepare_pika_event_broker():
    # mock a pika event broker
    pika_broker = Mock(spec=PikaEventBroker)

    # patch the spinner so we can execute the `_prepare_pika_producer()` function
    pika_broker.is_ready.return_value = True

    # noinspection PyProtectedMember
    export._prepare_event_broker(pika_broker)

    # the attributes are set as expected
    assert not pika_broker.should_keep_unpublished_messages
    assert pika_broker.raise_on_failure


@pytest.mark.parametrize(
    "current_timestamp,maximum_timestamp,endpoints_path,requested_ids,expected",
    [
        (1.0, None, None, None, "--minimum-timestamp 1.0"),
        (1.0, None, None, ["5", "6"], "--minimum-timestamp 1.0 --conversation-ids 5,6"),
        (1.0, 3.4, None, None, "--minimum-timestamp 1.0 --maximum-timestamp 3.4"),
        (
            1.0,
            2.5,
            "a.yml",
            None,
            "--endpoints a.yml --minimum-timestamp 1.0 --maximum-timestamp 2.5",
        ),
        (
            1.0,
            2.5,
            "a.yml",
            ["1", "2", "3"],
            (
                "--endpoints a.yml --minimum-timestamp 1.0 --maximum-timestamp 2.5 "
                "--conversation-ids 1,2,3"
            ),
        ),
    ],
)
def test_get_continuation_command(
    current_timestamp: float,
    maximum_timestamp: Optional[float],
    endpoints_path: Optional[Text],
    requested_ids: Optional[List[Text]],
    expected: Text,
):
    exporter = MockExporter()
    exporter.maximum_timestamp = maximum_timestamp
    exporter.endpoints_path = endpoints_path
    exporter.requested_conversation_ids = requested_ids

    # noinspection PyProtectedMember
    assert (
        export._get_continuation_command(exporter, current_timestamp)
        == f"rasa export {expected}"
    )


def _add_conversation_id_to_event(event: Dict, conversation_id: Text):
    event["sender_id"] = conversation_id


def prepare_namespace_and_mocked_tracker_store_with_events(
    temporary_path: Path, monkeypatch: MonkeyPatch
) -> Tuple[List[UserUttered], argparse.Namespace]:
    endpoints_path = write_endpoint_config_to_yaml(
        temporary_path,
        {"event_broker": {"type": "pika"}, "tracker_store": {"type": "sql"}},
    )

    # export these conversation IDs
    all_conversation_ids = ["id-1", "id-2", "id-3"]

    requested_conversation_ids = ["id-1", "id-2"]

    # create namespace with a set of cmdline arguments
    namespace = argparse.Namespace(
        endpoints=endpoints_path,
        conversation_ids=",".join(requested_conversation_ids),
        minimum_timestamp=1.0,
        maximum_timestamp=10.0,
    )

    # prepare events from different senders and different timestamps
    events = [random_user_uttered_event(timestamp) for timestamp in [1, 2, 3, 4, 11, 5]]
    events_for_conversation_id = {
        all_conversation_ids[0]: [events[0], events[1]],
        all_conversation_ids[1]: [events[2], events[3], events[4]],
        all_conversation_ids[2]: [events[5]],
    }

    def _get_tracker(conversation_id: Text) -> DialogueStateTracker:
        return DialogueStateTracker.from_events(
            conversation_id, events_for_conversation_id[conversation_id]
        )

    # mock tracker store
    tracker_store = Mock()
    tracker_store.keys.return_value = all_conversation_ids
    tracker_store.retrieve.side_effect = _get_tracker

    monkeypatch.setattr(export, "_get_tracker_store", lambda _: tracker_store)

    return events, namespace


def test_export_trackers(tmp_path: Path, monkeypatch: MonkeyPatch):
    events, namespace = prepare_namespace_and_mocked_tracker_store_with_events(
        tmp_path, monkeypatch
    )

    # mock event broker so we can check its `publish` method is called
    event_broker = Mock()
    event_broker.publish = Mock()
    monkeypatch.setattr(export, "_get_event_broker", lambda _: event_broker)

    # run the export function
    export.export_trackers(namespace)

    # check that only events 1, 2, 3, and 4 have been published
    # event 6 was sent by `id-3` which was not requested, and event 5
    # lies outside the requested time range
    calls = event_broker.publish.mock_calls

    # only four events were published (i.e. `publish()` method was called four times)
    assert len(calls) == 4

    # call objects are tuples of (name, pos. args, kwargs)
    # args itself is a tuple, and we want to access the first one, hence `call[1][0]`
    # check that events 1-4 were published
    assert all(
        any(call[1][0]["text"] == event.text for call in calls) for event in events[:4]
    )


@pytest.mark.parametrize("exception", [NoEventsToMigrateError, PublishingError(123)])
def test_export_trackers_publishing_exceptions(
    tmp_path: Path, monkeypatch: MonkeyPatch, exception: Exception
):
    events, namespace = prepare_namespace_and_mocked_tracker_store_with_events(
        tmp_path, monkeypatch
    )

    # mock event broker so we can check its `publish` method is called
    event_broker = Mock()
    event_broker.publish.side_effect = exception
    monkeypatch.setattr(export, "_get_event_broker", lambda _: event_broker)

    with pytest.raises(SystemExit):
        export.export_trackers(namespace)
