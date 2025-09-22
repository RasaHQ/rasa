import argparse
from pathlib import Path
from typing import Callable, List, Optional, Text, Tuple
from unittest.mock import AsyncMock, Mock

import pytest
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import RunResult

from rasa.cli import export
from rasa.core.brokers.broker import EventBroker
from rasa.core.brokers.pika import PikaEventBroker
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.config.configuration import Configuration
from rasa.exceptions import NoEventsToMigrateError, PublishingError
from rasa.shared.core.events import UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from tests.cli.conftest import RASA_EXE
from tests.conftest import (
    MockExporter,
    random_event,
    write_endpoint_config_to_yaml,
)


def test_export_help(run: Callable[..., RunResult]):
    output = run("export", "--help")

    help_text = f"""usage: {RASA_EXE} export [-h] [-v] [-vv] [--quiet]
                   [--logging-config-file LOGGING_CONFIG_FILE]
                   [--endpoints ENDPOINTS]
                   [--minimum-timestamp MINIMUM_TIMESTAMP]
                   [--maximum-timestamp MAXIMUM_TIMESTAMP]
                   [--offset-timestamps-by-seconds OFFSET_TIMESTAMPS_BY_SECONDS]
                   [--conversation-ids CONVERSATION_IDS]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


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
async def test_get_event_broker_and_tracker_store_from_endpoint_config(tmp_path: Path):
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

    # Clear the singleton instance of `AvailableEndpoints` to make sure we read the
    # endpoints from the test file.

    available_endpoints = Configuration.initialise_endpoints(
        endpoints_path=endpoints_path
    ).endpoints

    # fetching the event broker is successful
    assert await export._get_event_broker(available_endpoints)
    assert export._get_tracker_store(available_endpoints)


# noinspection PyProtectedMember
async def test_get_event_broker_from_endpoint_config_error_exit(tmp_path: Path):
    # write config without event broker to file
    endpoints_path = write_endpoint_config_to_yaml(
        tmp_path, {"tracker_store": {"type": "sql"}}
    )

    # Clear the singleton instance of `AvailableEndpoints` to make sure we read the
    # endpoints from the test file.

    available_endpoints = Configuration.initialise_endpoints(
        endpoints_path=endpoints_path
    ).endpoints

    with pytest.raises(SystemExit):
        assert await export._get_event_broker(available_endpoints)


def test_get_tracker_store_from_endpoint_config_error_exit(tmp_path: Path):
    # write config without event broker to file
    endpoints_path = write_endpoint_config_to_yaml(tmp_path, {})

    available_endpoints = Configuration.initialise_endpoints(
        endpoints_path=endpoints_path
    ).endpoints

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


def prepare_namespace_and_mocked_tracker_store_with_events(
    temporary_path: Path,
    monkeypatch: MonkeyPatch,
    is_pii_enabled: bool = False,
    is_event_anonymized: bool = False,
    request_extra_ids: bool = False,
) -> Tuple[List[UserUttered], argparse.Namespace]:
    config = {
        "event_broker": {"type": "pika"},
        "tracker_store": {"type": "sql"},
    }
    if is_pii_enabled:
        config["privacy"] = {
            "tracker_store_settings": {"anonymization": {"min_after_session_end": 120}},
            "rules": [{"slot": "slot_a", "anonymization": {"type": "mask"}}],
        }
    endpoints_path = write_endpoint_config_to_yaml(temporary_path, config)

    # export these conversation IDs
    all_conversation_ids = ["id-1", "id-2", "id-3", "id-4", "id-5"]

    requested_conversation_ids = ["id-1", "id-2"]

    if request_extra_ids:
        requested_conversation_ids.extend(["id-4", "id-5"])

    # create namespace with a set of cmdline arguments
    namespace = argparse.Namespace(
        endpoints=endpoints_path,
        conversation_ids=",".join(requested_conversation_ids),
        minimum_timestamp=1.0,
        maximum_timestamp=10.0,
        offset_timestamps_by_seconds=100,
    )

    # prepare events from different senders and different timestamps
    events = [
        random_event(timestamp, is_event_anonymized)
        for timestamp in [1, 2, 3, 4, 11, 5]
    ]
    events.append(random_event(6, is_event_anonymized, "bot"))
    events.append(random_event(7, is_event_anonymized, "slot"))
    events.append(random_event(8, is_event_anonymized, "action"))

    events_for_conversation_id = {
        all_conversation_ids[0]: [events[0], events[1]],
        all_conversation_ids[1]: [events[2], events[3], events[4]],
        all_conversation_ids[2]: [events[5]],
        all_conversation_ids[3]: [events[6], events[7]],
        all_conversation_ids[4]: [events[8]],
    }

    async def _get_tracker(conversation_id: Text) -> DialogueStateTracker:
        return DialogueStateTracker.from_events(
            conversation_id, events_for_conversation_id[conversation_id]
        )

    # mock tracker store
    tracker_store = Mock()
    tracker_store.keys = AsyncMock(return_value=all_conversation_ids)
    tracker_store.retrieve_full_tracker = _get_tracker

    monkeypatch.setattr(export, "_get_tracker_store", lambda _: tracker_store)

    return events, namespace


def test_export_trackers_with_offset(tmp_path: Path, monkeypatch: MonkeyPatch):
    events, namespace = prepare_namespace_and_mocked_tracker_store_with_events(
        tmp_path, monkeypatch
    )

    # mock event broker so we can check its `publish` method is called
    event_broker = Mock()

    async def _get_event_broker(_: AvailableEndpoints) -> EventBroker:
        return event_broker

    async def close():
        pass

    event_broker.close = close
    monkeypatch.setattr(export, "_get_event_broker", _get_event_broker)

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

    # check that the timestamps of the published events are offset by 100 seconds
    assert all(
        any(call[1][0]["timestamp"] == event.timestamp + 100 for call in calls)
        for event in events[:4]
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

    async def _get_event_broker(_: AvailableEndpoints) -> EventBroker:
        return event_broker

    monkeypatch.setattr(export, "_get_event_broker", _get_event_broker)

    with pytest.raises(SystemExit):
        export.export_trackers(namespace)


def test_rasa_export_unanonymized_events_warning_log(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture
):
    _, namespace = prepare_namespace_and_mocked_tracker_store_with_events(
        tmp_path,
        monkeypatch,
        is_pii_enabled=True,
        request_extra_ids=True,
    )

    # mock event broker
    event_broker = Mock()

    async def _get_event_broker(_: AvailableEndpoints) -> EventBroker:
        return event_broker

    async def close():
        pass

    event_broker.close = close
    monkeypatch.setattr(export, "_get_event_broker", _get_event_broker)

    # run the export function
    export.export_trackers(namespace)

    output = capsys.readouterr().out
    message = "Retrieved un-anonymized event for sender_id"

    assert f"{message} id-1" in output
    assert f"{message} id-2" in output
    assert f"{message} id-4" in output

    # id-5 is not a user, bot, or slot event, so it should not be logged
    assert f"{message} id-5" not in output


@pytest.mark.parametrize(
    "is_enabled, is_event_anonymized",
    [(True, True), (False, False)],
)
def test_rasa_export_no_unanonymized_events_warning_log(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture,
    is_enabled: bool,
    is_event_anonymized: bool,
):
    _, namespace = prepare_namespace_and_mocked_tracker_store_with_events(
        tmp_path,
        monkeypatch,
        is_pii_enabled=is_enabled,
        is_event_anonymized=is_event_anonymized,
    )

    # mock event broker
    event_broker = Mock()

    async def _get_event_broker(_: AvailableEndpoints) -> EventBroker:
        return event_broker

    async def close():
        pass

    event_broker.close = close
    monkeypatch.setattr(export, "_get_event_broker", _get_event_broker)

    # run the export function
    export.export_trackers(namespace)

    assert "Retrieved un-anonymized event for sender_id" not in capsys.readouterr().out
