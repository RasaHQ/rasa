import logging
import typing
from typing import Text

import rasa.shared.utils.common
from rasa.core.lock_store import LockStore

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa.core.agent import Agent

# backwards compatibility
run = rasa.run


def create_agent(model: Text, endpoints: Text = None) -> "Agent":
    """Create an agent instance based on a stored model.

    Args:
        model: file path to the stored model
        endpoints: file path to the used endpoint configuration
    """
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
