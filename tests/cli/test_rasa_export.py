import argparse
import random
import uuid
from pathlib import Path
from typing import Callable, Optional, Dict, Any, Text, List
from unittest.mock import Mock, patch

import pytest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import RunResult

import rasa.utils.io as io_utils
from rasa.cli import export
from rasa.core.brokers.pika import PikaEventBroker
from rasa.core.events import UserUttered
from rasa.core.tracker_store import TrackerStore
from rasa.core.trackers import DialogueStateTracker


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
def test_timestamps(
    minimum_timestamp: Optional[float], maximum_timestamp: Optional[float],
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
    export._inspect_timestamp_options(args)


@pytest.mark.parametrize(
    "minimum_timestamp,maximum_timestamp",
    [(3, 2), ("not-a-float", 5.5), (None, "no-float")],
)
def test_timestamp_error_exit(
    minimum_timestamp: Optional[float], maximum_timestamp: Optional[float],
):
    args = argparse.Namespace()
    args.minimum_timestamp = (
        str(minimum_timestamp) if minimum_timestamp is not None else None
    )
    args.maximum_timestamp = (
        str(maximum_timestamp) if maximum_timestamp is not None else None
    )

    with pytest.raises(SystemExit):
        # noinspection PyProtectedMember
        export._inspect_timestamp_options(args)


def _write_endpoint_config_to_yaml(path: Path, data: Dict[Text, Any]) -> Path:
    endpoints_path = path / "endpoints.yml"

    # write endpoints config to file
    io_utils.write_yaml_file(
        data, endpoints_path,
    )
    return endpoints_path


def test_get_available_endpoints(tmp_path: Path):
    # write valid config to file
    endpoints_path = _write_endpoint_config_to_yaml(
        tmp_path, {"event_broker": {"type": "pika"}, "tracker_store": {"type": "sql"}}
    )

    # noinspection PyProtectedMember
    available_endpoints = export._get_available_endpoints(str(endpoints_path))

    # assert event broker and tracker store are valid, others are not
    assert available_endpoints.tracker_store and available_endpoints.event_broker
    assert not available_endpoints.lock_store and not available_endpoints.nlg


# noinspection PyProtectedMember
def test_get_event_broker_and_tracker_store_from_endpoint_config(tmp_path: Path):
    # write valid config to file
    endpoints_path = _write_endpoint_config_to_yaml(
        tmp_path, {"event_broker": {"type": "sql"}, "tracker_store": {"type": "sql"}},
    )

    available_endpoints = export._get_available_endpoints(str(endpoints_path))

    # fetching the event broker is successful
    assert export._get_event_broker(available_endpoints)
    assert export._get_rasa_tracker_store(available_endpoints)


# noinspection PyProtectedMember
def test_get_event_broker_from_endpoint_config_error_exit(tmp_path: Path):
    # write config without event broker to file
    endpoints_path = _write_endpoint_config_to_yaml(
        tmp_path, {"tracker_store": {"type": "sql"}}
    )

    available_endpoints = export._get_available_endpoints(str(endpoints_path))

    with pytest.raises(SystemExit):
        assert export._get_event_broker(available_endpoints)


# noinspection PyProtectedMember
def test_get_tracker_store_from_endpoint_config_error_exit(tmp_path: Path):
    # write config without event broker to file
    endpoints_path = _write_endpoint_config_to_yaml(tmp_path, {})

    available_endpoints = export._get_available_endpoints(str(endpoints_path))

    with pytest.raises(SystemExit):
        assert export._get_rasa_tracker_store(available_endpoints)


@pytest.mark.parametrize(
    "requested_ids,expected",
    [("id1", ["id1"]), ("id1,id2", ["id1", "id2"]), (None, None)],
)
def test_get_requested_conversation_ids(
    requested_ids: Optional[Text], expected: Optional[List[Text]]
):
    # noinspection PyProtectedMember
    assert export._get_requested_conversation_ids(requested_ids) == expected


@pytest.mark.parametrize(
    "requested_ids,available_ids,expected",
    [([1], [1], [1]), ([1, 2], [2], [2]), (None, [2], [2])],
)
def test_get_conversation_ids_to_process(
    requested_ids: Optional[List[int]],
    available_ids: Optional[List[int]],
    expected: Optional[List[int]],
    monkeypatch: MonkeyPatch,
):
    # convert ids to strings
    _requested_ids = [str(_id) for _id in requested_ids] if requested_ids else None
    _available_ids = [str(_id) for _id in available_ids]
    _expected = [str(_id) for _id in expected] if expected else None

    # create and mock tracker store contain `available_ids` as keys
    tracker_store = TrackerStore(None, None)
    monkeypatch.setattr(tracker_store, "keys", lambda: _available_ids)

    # noinspection PyProtectedMember
    assert (
        export._get_conversation_ids_to_process(tracker_store, _requested_ids)
        == _expected
    )


@pytest.mark.parametrize(
    "requested_ids,available_ids",
    [
        ([1], []),  # no IDs in tracker store
        (None, []),  # same thing, but without requested IDs
        ([1, 2, 3], [4, 5, 6]),  # no overlap between requested IDs and those available
    ],
)
def test_get_conversation_ids_to_process_error_exit(
    requested_ids: Optional[List[int]],
    available_ids: Optional[List[int]],
    monkeypatch: MonkeyPatch,
):
    # convert ids to strings
    _requested_ids = [str(_id) for _id in requested_ids] if requested_ids else None
    _available_ids = [str(_id) for _id in available_ids]

    # create and mock tracker store contain `available_ids` as keys
    tracker_store = TrackerStore(None, None)
    monkeypatch.setattr(tracker_store, "keys", lambda: _available_ids)

    with pytest.raises(SystemExit):
        # noinspection PyProtectedMember
        export._get_conversation_ids_to_process(tracker_store, _requested_ids)


def test_prepare_pika_event_broker():
    # mock a pika event broker
    pika_broker = Mock(spec=PikaEventBroker)

    # patch the spinner so we can execute the `_prepare_pika_producer()` function
    with patch.object(export, "_ensure_pika_channel_is_open", lambda _: None):
        # noinspection PyProtectedMember
        export._prepare_pika_producer(pika_broker)

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
    # noinspection PyProtectedMember
    assert (
        export._get_continuation_command(
            current_timestamp, maximum_timestamp, endpoints_path, requested_ids
        )
        == f"rasa export {expected}"
    )


def random_user_uttered_event(timestamp: Optional[float] = None) -> UserUttered:
    return UserUttered(
        uuid.uuid4().hex,
        timestamp=timestamp if timestamp is not None else random.random(),
    )


# noinspection PyProtectedMember
def test_fetch_events_within_time_range():
    conversation_ids = ["some-id", "another-id"]

    # prepare events with from different senders and different timestamps
    event_1 = random_user_uttered_event(3)
    event_2 = random_user_uttered_event(2)
    event_3 = random_user_uttered_event(1)
    events = {
        conversation_ids[0]: [event_1, event_2],
        conversation_ids[1]: [event_3],
    }

    def _get_tracker(conversation_id: Text) -> DialogueStateTracker:
        return DialogueStateTracker.from_events(
            conversation_id, events[conversation_id]
        )

    # create mock tracker store
    tracker_store = Mock(spec=TrackerStore)
    tracker_store.retrieve.side_effect = _get_tracker

    fetched_events = export._fetch_events_within_time_range(
        tracker_store, None, None, conversation_ids
    )

    # events should come back for all requested conversation IDs
    assert all(
        any(_id in event["sender_id"] for event in fetched_events)
        for _id in conversation_ids
    )

    # events are sorted by timestamp despite the initially different order
    assert fetched_events == list(sorted(fetched_events, key=lambda e: e["timestamp"]))


# noinspection PyProtectedMember
def test_sort_and_select_events_by_timestamp():
    events = [
        event.as_dict()
        for event in [
            random_user_uttered_event(3),
            random_user_uttered_event(2),
            random_user_uttered_event(1),
        ]
    ]

    selected_events = export._sort_and_select_events_by_timestamp(events)

    # events are sorted
    assert selected_events == list(
        sorted(selected_events, key=lambda e: e["timestamp"])
    )

    # apply minimum timestamp requirement, expect to get only two events back
    assert export._sort_and_select_events_by_timestamp(
        events, minimum_timestamp=2.0
    ) == [events[1], events[0]]

    # apply maximum timestamp requirement, expect to get only one
    assert export._sort_and_select_events_by_timestamp(
        events, maximum_timestamp=1.1
    ) == [events[2]]

    # apply both requirements, get one event back
    assert export._sort_and_select_events_by_timestamp(
        events, minimum_timestamp=2.0, maximum_timestamp=2.1
    ) == [events[1]]


# noinspection PyProtectedMember
def test_sort_and_select_events_by_timestamp_error_exit():
    # no events given
    with pytest.raises(SystemExit):
        export._sort_and_select_events_by_timestamp([])

    # supply list of events, apply timestamp constraint and no events survive
    events = [random_user_uttered_event(3).as_dict()]
    with pytest.raises(SystemExit):
        export._sort_and_select_events_by_timestamp(events, minimum_timestamp=3.1)
