import logging
import os
import shutil
import typing
from typing import Dict, Text

from rasa.constants import DOCS_BASE_URL
from rasa.cli.utils import minimal_kwargs, print_warning, print_error
from rasa.model import get_model

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa.core.agent import Agent


def run(
    model: Text,
    endpoints: Text,
    connector: Text = None,
    credentials: Text = None,
    **kwargs: Dict
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

    model_path = get_model(model)

    _endpoints = AvailableEndpoints.read_endpoints(endpoints)

    if not connector and not credentials:
        connector = "rest"
        print_warning(
            "No chat connector configured, falling back to the "
            "REST input channel. To connect your bot to another channel, "
            "read the docs here: {}/user-guide/"
            "messaging-and-voice-channels".format(DOCS_BASE_URL)
        )

    kwargs = minimal_kwargs(kwargs, rasa.core.run.serve_application)
    rasa.core.run.serve_application(
        model,
        channel=connector,
        credentials=credentials,
        endpoints=_endpoints,
        **kwargs
    )

    if model_path is not None:
        shutil.rmtree(model_path)


def create_agent(model: Text, endpoints: Text = None) -> "Agent":
    from rasa.core.tracker_store import TrackerStore
    from rasa.core import broker
    from rasa.core.utils import AvailableEndpoints

    _endpoints = AvailableEndpoints.read_endpoints(endpoints)

    _broker = broker.from_endpoint_config(_endpoints.event_broker)

    _tracker_store = TrackerStore.find_tracker_store(
        None, _endpoints.tracker_store, _broker
    )

    return Agent.load(
        model,
        generator=_endpoints.nlg,
        tracker_store=_tracker_store,
        action_endpoint=_endpoints.action,
    )
