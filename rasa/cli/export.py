import argparse
import logging
import typing
from typing import List, Text, Optional

from rasa import telemetry
from rasa.cli import SubParsersAction
import rasa.core.utils
import rasa.shared.utils.cli
import rasa.utils.common
from rasa.cli.arguments import export as arguments
from rasa.shared.constants import DOCS_URL_EVENT_BROKERS, DOCS_URL_TRACKER_STORES
from rasa.exceptions import PublishingError
from rasa.shared.exceptions import RasaException
from rasa.core.brokers.pika import PikaEventBroker

if typing.TYPE_CHECKING:
    from rasa.core.brokers.broker import EventBroker
    from rasa.core.tracker_store import TrackerStore
    from rasa.core.exporter import Exporter
    from rasa.core.utils import AvailableEndpoints

logger = logging.getLogger(__name__)


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add subparser for `rasa export`.

    Args:
        subparsers: Subparsers action object to which `argparse.ArgumentParser`
            objects can be added.
        parents: `argparse.ArgumentParser` objects whose arguments should also be
            included.
    """
    export_parser = subparsers.add_parser(
        "export",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Export conversations using an event broker.",
    )
    export_parser.set_defaults(func=export_trackers)

    arguments.set_export_arguments(export_parser)


def _get_tracker_store(endpoints: "AvailableEndpoints") -> "TrackerStore":
    """Get `TrackerStore` from `endpoints`.

    Prints an error and exits if no tracker store could be loaded.

    Args:
        endpoints: `AvailableEndpoints` to initialize the tracker store from.

    Returns:
        Initialized tracker store.

    """
    if not endpoints.tracker_store:
        rasa.shared.utils.cli.print_error_and_exit(
            f"Could not find a `tracker_store` section in the supplied "
            f"endpoints file. Instructions on how to configure a tracker store "
            f"can be found here: {DOCS_URL_TRACKER_STORES}. "
            f"Exiting. "
        )

    from rasa.core.tracker_store import TrackerStore

    return TrackerStore.create(endpoints.tracker_store)


async def _get_event_broker(endpoints: "AvailableEndpoints") -> "EventBroker":
    """Get `EventBroker` from `endpoints`.

    Prints an error and exits if no event broker could be loaded.

    Args:
        endpoints: `AvailableEndpoints` to initialize the event broker from.

    Returns:
        Initialized event broker.

    """
    from rasa.core.brokers.broker import EventBroker

    broker = await EventBroker.create(endpoints.event_broker)

    if not broker:
        rasa.shared.utils.cli.print_error_and_exit(
            f"Could not find an `event_broker` section in the supplied "
            f"endpoints file. Instructions on how to configure an event broker "
            f"can be found here: {DOCS_URL_EVENT_BROKERS}. Exiting."
        )
    return broker


def _get_requested_conversation_ids(
    conversation_ids_arg: Optional[Text] = None,
) -> Optional[List[Text]]:
    """Get list of conversation IDs requested as a command-line argument.

    Args:
        conversation_ids_arg: Value of `--conversation-ids` command-line argument.
            If provided, this is a string of comma-separated conversation IDs.

    Return:
        List of conversation IDs requested as a command-line argument.
        `None` if that argument was left unspecified.

    """
    if not conversation_ids_arg:
        return None

    return conversation_ids_arg.split(",")


def _assert_max_timestamp_is_greater_than_min_timestamp(
    args: argparse.Namespace,
) -> None:
    """Inspect CLI timestamp parameters.

    Prints an error and exits if a maximum timestamp is provided that is smaller
    than the provided minimum timestamp.

    Args:
        args: Command-line arguments to process.

    """
    min_timestamp = args.minimum_timestamp
    max_timestamp = args.maximum_timestamp

    if (
        min_timestamp is not None
        and max_timestamp is not None
        and max_timestamp < min_timestamp
    ):
        rasa.shared.utils.cli.print_error_and_exit(
            f"Maximum timestamp '{max_timestamp}' is smaller than minimum "
            f"timestamp '{min_timestamp}'. Exiting."
        )


def _prepare_event_broker(event_broker: "EventBroker") -> None:
    """Prepares event broker to export tracker events.

    Sets `should_keep_unpublished_messages` flag to `False` if
    `self.event_broker` is a `PikaEventBroker`.

    If publishing of events fails, the `PikaEventBroker` instance should not keep a
    list of unpublished messages, so we can retry publishing them. This is because
    the instance is launched as part of this short-lived export script, meaning the
    object is destroyed before it might be published.

    In addition, wait until the event broker reports a `ready` state.

    """
    if isinstance(event_broker, PikaEventBroker):
        event_broker.should_keep_unpublished_messages = False
        event_broker.raise_on_failure = True

    if not event_broker.is_ready():
        rasa.shared.utils.cli.print_error_and_exit(
            f"Event broker of type '{type(event_broker)}' is not ready. Exiting."
        )


def export_trackers(args: argparse.Namespace) -> None:
    """Export events for a connected tracker store using an event broker.

    Args:
        args: Command-line arguments to process.
    """
    rasa.utils.common.run_in_loop(_export_trackers(args))


async def _export_trackers(args: argparse.Namespace) -> None:

    _assert_max_timestamp_is_greater_than_min_timestamp(args)

    endpoints = rasa.core.utils.read_endpoints_from_path(args.endpoints)
    tracker_store = _get_tracker_store(endpoints)
    event_broker = await _get_event_broker(endpoints)
    _prepare_event_broker(event_broker)
    requested_conversation_ids = _get_requested_conversation_ids(args.conversation_ids)

    from rasa.core.exporter import Exporter

    exporter = Exporter(
        tracker_store,
        event_broker,
        args.endpoints,
        requested_conversation_ids,
        args.minimum_timestamp,
        args.maximum_timestamp,
    )

    try:
        published_events = await exporter.publish_events()
        telemetry.track_tracker_export(published_events, tracker_store, event_broker)
        rasa.shared.utils.cli.print_success(
            f"Done! Successfully published {published_events} events 🎉"
        )

    except PublishingError as e:
        command = _get_continuation_command(exporter, e.timestamp)
        rasa.shared.utils.cli.print_error_and_exit(
            f"Encountered error while publishing event with timestamp '{e}'. To "
            f"continue where I left off, run the following command:"
            f"\n\n\t{command}\n\nExiting."
        )

    except RasaException as e:
        rasa.shared.utils.cli.print_error_and_exit(str(e))


def _get_continuation_command(exporter: "Exporter", timestamp: float) -> Text:
    """Build CLI command to continue 'rasa export' where it was interrupted.

    Called when event publishing stops due to an error.

    Args:
        exporter: Exporter object containing objects relevant for this export.
        timestamp: Timestamp of the last event attempted to be published.

    """
    # build CLI command command based on supplied timestamp and options
    command = "rasa export"

    if exporter.endpoints_path is not None:
        command += f" --endpoints {exporter.endpoints_path}"

    command += f" --minimum-timestamp {timestamp}"

    if exporter.maximum_timestamp is not None:
        command += f" --maximum-timestamp {exporter.maximum_timestamp}"

    if exporter.requested_conversation_ids:
        command += (
            f" --conversation-ids {','.join(exporter.requested_conversation_ids)}"
        )

    return command
