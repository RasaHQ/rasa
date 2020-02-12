import argparse
import itertools
import logging
import time
from typing import (
    List,
    Text,
    Optional,
    Dict,
    Any,
)

from tqdm import tqdm

import rasa.cli.utils as cli_utils
from rasa.cli.arguments import export as arguments
from rasa.constants import DEFAULT_ENDPOINTS_PATH
from rasa.core.brokers.broker import EventBroker
from rasa.core.brokers.pika import PikaEventBroker
from rasa.core.tracker_store import TrackerStore
from rasa.core.trackers import EventVerbosity
from rasa.core.utils import AvailableEndpoints

logger = logging.getLogger(__name__)


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    export_parser_args = {
        "parents": parents,
        "conflict_handler": "resolve",
        "formatter_class": argparse.ArgumentDefaultsHelpFormatter,
        "help": "Export Rasa trackers using an event broker.",
    }

    shell_parser = subparsers.add_parser("export", **export_parser_args)
    shell_parser.set_defaults(func=export_trackers)

    arguments.set_export_arguments(shell_parser)


def _get_rasa_tracker_store(endpoints: AvailableEndpoints) -> TrackerStore:
    """Get tracker store from `endpoints`.

    Prints an error and exits if no tracker store could be loaded.

    Args:
        endpoints: `AvailableEndpoints` to initialize the tracker store from.

    Returns:
        Initialized tracker store.

    """
    if not endpoints.tracker_store:
        cli_utils.print_error_and_exit(
            "Could not find a `tracker_store` section in the supplied "
            "endpoints file. Exiting."
        )

    return TrackerStore.create(endpoints.tracker_store)


def _get_event_broker(endpoints: AvailableEndpoints) -> Optional[EventBroker]:
    """Get event broker from `endpoints`.

    Prints an error and exits if no event broker could be loaded.

    Args:
        endpoints: `AvailableEndpoints` to initialize the event broker from.

    Returns:
        Initialized event broker.

    """
    if not endpoints.event_broker:
        cli_utils.print_error_and_exit(
            "Could not find an `event_broker` section in the supplied "
            "endpoints file. Exiting."
        )

    return EventBroker.create(endpoints.event_broker)


def _get_available_endpoints(endpoints_path: Optional[Text]) -> AvailableEndpoints:
    """Get `AvailableEndpoints` object from specified path.

    Args:
        endpoints_path: Path of the endpoints file to be read. If `None` the
            default path for that file is used (`endpoints.yml`).

    Returns:
        `AvailableEndpoints` object read from endpoints file.

    """
    endpoints_config_path = cli_utils.get_validated_path(
        endpoints_path, "endpoints", DEFAULT_ENDPOINTS_PATH, True
    )
    return AvailableEndpoints.read_endpoints(endpoints_config_path)


def _get_requested_conversation_ids(
    conversation_ids_arg: Optional[Text] = None,
) -> Optional[List[Text]]:
    """Get list of conversation IDs requested as a command-line argument.

    Args:
        conversation_ids_arg: Value of `--conversation-ids` command-line argument. If
            provided, this is a string of comma-separated conversation IDs.

    Return:
        List of conversation IDs requested as a command-line argument. `None` if that
        argument was left unspecified.

    """
    if not conversation_ids_arg:
        return None

    return conversation_ids_arg.split(",")


def _get_conversation_ids_to_process(
    tracker_store: TrackerStore,
    requested_conversation_ids: Optional[List[Text]] = None,
) -> List[Text]:
    """Get conversation IDs that are good for processing.

    Finds the intersection of events that are contained in the tracker store with
    those events requested as a command-line argument.

    Prints an error and if no conversation IDs are found in the tracker, or if no
    overlap is found between those contained in the tracker and those requested
    by the user.

    Args:
        tracker_store: Tracker store to source events from.
        requested_conversation_ids: List of conversation IDs that should be published
            requested by the user. If `None`, all conversation IDs contained in the
            tracker store are published.

    Returns:
        Conversation IDs that are both requested and contained in the tracker
        store. If no conversation IDs are requested, all conversation IDs in the
        tracker store are returned.

    """
    conversation_ids_in_tracker_store = list(tracker_store.keys())

    if not conversation_ids_in_tracker_store:
        cli_utils.print_error_and_exit(
            f"Could not find any conversations in connected tracker store. Exiting."
        )

    if not requested_conversation_ids:
        return conversation_ids_in_tracker_store

    missing_ids_in_tracker_store = set(requested_conversation_ids) - set(
        conversation_ids_in_tracker_store
    )

    if missing_ids_in_tracker_store:
        cli_utils.print_warning(
            f"Could not find the following requested "
            f"conversation IDs in connected tracker store: "
            f"{', '.join(sorted(missing_ids_in_tracker_store))}"
        )

    conversation_ids_to_process = set(conversation_ids_in_tracker_store) & set(
        requested_conversation_ids
    )

    if not conversation_ids_to_process:
        cli_utils.print_error_and_exit(
            "Could not find an overlap between the requested "
            "conversation IDs and those found in the tracker store. Exiting."
        )

    return list(conversation_ids_to_process)


def _inspect_timestamp_options(args: argparse.Namespace) -> None:
    """Inspect CLI timestamp parameters.

    Prints an error and exits if the supplied timestamp parameters cannot be
    converted to `float`, or if a maximum timestamp is provided that is smaller
    than the provided minimum timestamp.

    Args:
        args: Command-line arguments to process.

    """
    # do nothing if no timestamp CLI arguments are provided
    if args.minimum_timestamp is None and args.maximum_timestamp is None:
        return

    min_timestamp, max_timestamp = None, None

    if args.minimum_timestamp is not None:
        try:
            min_timestamp = float(args.minimum_timestamp)
        except (TypeError, SyntaxError, ValueError):
            cli_utils.print_error_and_exit(
                f"Failed to convert minimum timestamp parameter "
                f"'{args.minimum_timestamp}' to `float`."
            )

    if args.maximum_timestamp is not None:
        try:
            max_timestamp = float(args.maximum_timestamp)
        except (TypeError, SyntaxError, ValueError):
            cli_utils.print_error_and_exit(
                f"Failed to convert maximum timestamp parameter "
                f"'{args.maximum_timestamp}' to `float`."
            )

    if args.minimum_timestamp is not None and args.maximum_timestamp is not None:
        if max_timestamp < min_timestamp:
            cli_utils.print_error_and_exit(
                f"Maximum timestamp '{max_timestamp}' is smaller than minimum "
                f"timestamp '{min_timestamp}'. Exiting."
            )


def _ensure_pika_channel_is_open(
    event_broker: PikaEventBroker,
    attempts: int = 1000,
    wait_time_between_attempts: float = 0.01,
) -> None:
    """Spin until the pika channel is open.

    Prints an error and exits if the channel does not open.

    It typically takes 50 ms or so for that to happen. We'll wait up to 10
    seconds just in case.

    Args:
        event_broker: Pika event broker to wait for.
        attempts: Number of retries.
        wait_time_between_attempts: Wait time between retries.

    """
    while attempts:
        if event_broker.channel:
            return
        time.sleep(wait_time_between_attempts)
        attempts -= 1

    cli_utils.print_error_and_exit("Failed to open Pika channel. Exiting.")


def _prepare_pika_producer(event_broker: EventBroker) -> None:
    """Sets `should_keep_unpublished_messages` flag to `False` if `event_broker`
    is a `PikaEventBroker`.

    If publishing of events fails, the `PikaEventBroker` instance should not keep a
    list of unpublished messages, so we can retry publishing them. This is because
    the instance is launched as part of this short-lived export script, meaning the
    object is destroyed before it might be published.


    Args:
        event_broker: Event broker to modify if it's a `PikaEventBroker`.

    """
    if isinstance(event_broker, PikaEventBroker):
        event_broker.should_keep_unpublished_messages = False
        event_broker.raise_on_failure = True

        # the pika channel takes a short while to open, so we'll spin until that happens
        _ensure_pika_channel_is_open(event_broker)


def export_trackers(args: argparse.Namespace) -> None:
    """Export events for a connected tracker store using an event broker.

    Args:
        args: Command-line arguments to process.

    """
    _inspect_timestamp_options(args)

    endpoints = _get_available_endpoints(args.endpoints)
    rasa_tracker_store = _get_rasa_tracker_store(endpoints)
    event_broker = _get_event_broker(endpoints)
    _prepare_pika_producer(event_broker)

    requested_conversation_ids = _get_requested_conversation_ids(args.conversation_ids)
    conversation_ids_to_process = _get_conversation_ids_to_process(
        rasa_tracker_store, requested_conversation_ids
    )

    _publish_events(
        rasa_tracker_store,
        event_broker,
        conversation_ids_to_process,
        args.minimum_timestamp,
        args.maximum_timestamp,
        args.endpoints,
        requested_conversation_ids,
    )


def _sort_and_select_events_by_timestamp(
    events: List[Dict[Text, Any]],
    minimum_timestamp: Optional[float] = None,
    maximum_timestamp: Optional[float] = None,
) -> List[Dict[Text, Any]]:
    """Sort list of events by ascending timestamp, and select events within time range.

    Prints an error message and exits if no events are found within the requested
    time range.

    Args:
        events: List of serialized events to be sorted and selected from.
        minimum_timestamp: Minimum timestamp of events that are published. If `None`,
            apply no such constraint.
        maximum_timestamp: Maximum timestamp of events that are published. If `None`,
            apply no such constraint.

    Returns:
        List of serialized and sorted (by timestamp) events within the requested time
        range.

    """
    cli_utils.print_info(
        f"Sorting and selecting from {len(events)} total events found."
    )
    # sort the events by timestamp just in case they're not sorted already
    events = sorted(events, key=lambda x: x["timestamp"])

    # drop events failing minimum timestamp requirement
    if minimum_timestamp is not None:
        events = itertools.dropwhile(
            lambda x: x["timestamp"] < minimum_timestamp, events
        )

    # select events passing maximum timestamp requirement
    if maximum_timestamp is not None:
        events = itertools.takewhile(
            lambda x: x["timestamp"] < maximum_timestamp, events
        )

    events = list(events)
    if not events:
        cli_utils.print_error_and_exit(
            "Could not find any events within requested time range. Exiting."
        )

    return events


def _fetch_events_within_time_range(
    tracker_store: TrackerStore,
    minimum_timestamp: Optional[float] = None,
    maximum_timestamp: Optional[float] = None,
    conversation_ids: Optional[List[Text]] = None,
) -> List[Dict[Text, Any]]:
    """Fetch all events for `conversation_ids` within the supplied time range.

    Args:
        tracker_store: Tracker store to source events from.
        minimum_timestamp: Minimum timestamp of events that are published. If `None`,
            apply no such constraint.
        maximum_timestamp: Maximum timestamp of events that are published. If `None`,
            apply no such constraint.
         conversation_ids: List of conversation IDs selected for publishing.

    Returns:
        List of serialized events with added `sender_id` field.

    """
    cli_utils.print_info(
        f"Fetching events for {len(conversation_ids)} conversation IDs:"
    )

    events = []

    for conversation_id in tqdm(conversation_ids, "conversation IDs"):
        tracker = tracker_store.retrieve(conversation_id)
        if not tracker:
            logger.info(
                f"Could not retrieve tracker for conversation ID "
                f"'{conversation_id}'. Skipping."
            )
            continue

        _events = tracker.current_state(EventVerbosity.ALL)["events"]

        if not _events:
            logger.info(
                f"No events to migrate for conversation ID '{conversation_id}'."
            )

        # the conversation IDs are needed in the event publishing
        for event in _events:
            event["sender_id"] = conversation_id
            events.append(event)

    return _sort_and_select_events_by_timestamp(
        events, minimum_timestamp, maximum_timestamp
    )


def _get_continuation_command(
    timestamp: float,
    maximum_timestamp: Optional[float] = None,
    endpoints_path: Optional[Text] = None,
    requested_conversation_ids: Optional[List[Text]] = None,
) -> Text:
    """Build CLI command to continue 'rasa export' where it was interrupted.

    Called when event publishing stops due to an error.

    Args:
        timestamp: Timestamp of the last event attempted to be published.
        maximum_timestamp: Maximum timestamp of events that are published. If `None`,
            apply no such constraint.
        endpoints_path: Path to the endpoints file used to configure the event broker
            and tracker store. If `None`, the default path ('endpoints.yml') is used.
        requested_conversation_ids: List of conversation IDs that should be published
            requested by the user. If `None`, all conversation IDs contained in the
            tracker store are published.

    """
    # build CLI command command based on supplied timestamp and options
    command = f"rasa export"

    if endpoints_path is not None:
        command += f" --endpoints {endpoints_path}"

    command += f" --minimum-timestamp {timestamp}"

    if maximum_timestamp is not None:
        command += f" --maximum-timestamp {maximum_timestamp}"

    if requested_conversation_ids:
        command += f" --conversation-ids {','.join(requested_conversation_ids)}"

    return command


def _publish_events(
    tracker_store: TrackerStore,
    event_broker: EventBroker,
    conversation_ids: List[Text],
    minimum_timestamp: Optional[float] = None,
    maximum_timestamp: Optional[float] = None,
    endpoints_path: Optional[Text] = None,
    requested_conversation_ids: Optional[List[Text]] = None,
) -> None:
    """Publish events in a tracker store using an event broker.

    Exits if the publishing of events is interrupted due to an error. In that case,
    the CLI command to continue the export where it was interrupted is printed.

    Args:
        tracker_store: Tracker store to source events from.
        event_broker: Event broker used to publish events over.
        conversation_ids: List of conversation IDs selected for publishing.
        minimum_timestamp: Minimum timestamp of events that are published. If `None`,
            apply no such constraint.
        maximum_timestamp: Maximum timestamp of events that are published. If `None`,
            apply no such constraint.
        endpoints_path: Path to the endpoints file used to configure the event broker
            and tracker store. If `None`, the default path ('endpoints.yml') is used.
        requested_conversation_ids: List of conversation IDs that should be published
            requested by the user. If `None`, all conversation IDs contained in the
            tracker store are published.

    """
    events = _fetch_events_within_time_range(
        tracker_store, minimum_timestamp, maximum_timestamp, conversation_ids
    )

    cli_utils.print_info(f"Selected {len(events)} events. Ready to publish.")

    published_events = 0
    current_timestamp = None

    for event in tqdm(events, "events"):
        # noinspection PyBroadException
        try:
            body = {"sender_id": event["sender_id"]}
            body.update(event)
            event_broker.publish(body)
            published_events += 1
            current_timestamp = event["timestamp"]
        except Exception as e:
            logger.exception(e)

            command = _get_continuation_command(
                current_timestamp,
                maximum_timestamp,
                endpoints_path,
                requested_conversation_ids,
            )
            cli_utils.print_error_and_exit(
                f"Encountered error while publishing event "
                f"{published_events}/{len(events)}. To continue where I left off, "
                f"run the following command:\n\n\t{command}\n\nExiting."
            )

    cli_utils.print_success(
        f"Done! Successfully published '{published_events}' events 🎉"
    )
