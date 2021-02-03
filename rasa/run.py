import logging
import typing
from typing import Dict, Text

import rasa.shared.utils.common
from rasa.shared.utils.cli import print_warning
from rasa.shared.constants import DOCS_BASE_URL
from rasa.core.lock_store import LockStore

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa.core.agent import Agent


def run(
    model: Text,
    endpoints: Text,
    connector: Text = None,
    credentials: Text = None,
    **kwargs: Dict,
):
    """Runs a Rasa model.

    Args:
        model: Path to model archive.
        endpoints: Path to endpoints file.
        connector: Connector which should be use (overwrites `credentials`
        field).
        credentials: Path to channel credentials file.
        **kwargs: Additional arguments which are passed to
        `rasa.core.run.serve_application`.

    """
    import rasa.core.run
    import rasa.nlu.run
    from rasa.core.utils import AvailableEndpoints

    _endpoints = AvailableEndpoints.read_endpoints(endpoints)

    if not connector and not credentials:
        connector = "rest"
        print_warning(
            f"No chat connector configured, falling back to the "
            f"REST input channel. To connect your bot to another channel, "
            f"read the docs here: {DOCS_BASE_URL}/messaging-and-voice-channels"
        )

    kwargs = rasa.shared.utils.common.minimal_kwargs(
        kwargs, rasa.core.run.serve_application
    )
    rasa.core.run.serve_application(
        model,
        channel=connector,
        credentials=credentials,
        endpoints=_endpoints,
        **kwargs,
    )


def create_agent(model: Text, endpoints: Text = None) -> "Agent":
    from rasa.core.tracker_store import TrackerStore
    from rasa.core.utils import AvailableEndpoints
    from rasa.core.agent import Agent
    from rasa.core.brokers.broker import EventBroker
    import rasa.utils.common

    _endpoints = AvailableEndpoints.read_endpoints(endpoints)

    _broker = rasa.utils.common.run_in_loop(EventBroker.create(_endpoints.event_broker))
    _tracker_store = TrackerStore.create(_endpoints.tracker_store, event_broker=_broker)
    _lock_store = LockStore.create(_endpoints.lock_store)

    return Agent.load(
        model,
        generator=_endpoints.nlg,
        tracker_store=_tracker_store,
        lock_store=_lock_store,
        action_endpoint=_endpoints.action,
    )
