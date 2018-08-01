from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import logging
import os
import time

import requests
from future.moves.urllib.parse import quote_plus
from requests.exceptions import RequestException
from six import string_types
from typing import Callable, Union
from typing import Text, List, Optional, Dict, Any

from rasa_core import utils
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.channels import InputChannel
from rasa_core.channels import UserMessage
from rasa_core.dispatcher import Dispatcher
from rasa_core.domain import Domain, TemplateDomain
from rasa_core.events import BotUttered
from rasa_core.events import Event
from rasa_core.nlg import NaturalLanguageGenerator
from rasa_core.trackers import DialogueStateTracker
from rasa_core.utils import EndpointConfig

logger = logging.getLogger(__name__)


class RasaCoreClient(object):
    """Connects to a running Rasa Core server.

    Used to retrieve information about models and conversations."""

    def __init__(self, core_endpoint):
        # type: (EndpointConfig) -> None

        self.core_endpoint = core_endpoint

    def status(self):
        """Get the status of the remote core server (e.g. the version.)"""

        result = self.core_endpoint.request(subpath="/version", method="get")
        result.raise_for_status()
        return result.json()

    def clients(self):
        """Get a list of all conversations."""

        result = self.core_endpoint.request(subpath="/conversations",
                                            method="get")
        result.raise_for_status()
        return result.json()

    def tracker(self,
                sender_id,  # type: Text
                domain,  # type: Domain
                only_events_after_latest_restart=False,  # type: bool
                include_events=True,  # type: bool
                until=None  # type: Optional[int]
                ):
        """Retrieve and recreate a tracker fetched from the remote instance."""

        tracker_json = self.tracker_json(
                sender_id, only_events_after_latest_restart,
                include_events, until)

        tracker = DialogueStateTracker.from_dict(
                sender_id, tracker_json.get("events", []), domain)
        return tracker

    def tracker_json(self,
                     sender_id,  # type: Text
                     use_history=True,  # type: bool
                     include_events=True,  # type: bool
                     until=None  # type: Optional[int]
                     ):
        """Retrieve a tracker's json representation from remote instance."""

        url = "/conversations/{}/tracker?ignore_restarts={}&events={}".format(
                sender_id, use_history, include_events)
        if until:
            url += "&until={}".format(until)

        result = self.core_endpoint.request(subpath=url,
                                            method="get")
        result.raise_for_status()
        return result.json()

    def append_events_to_tracker(self, sender_id, events):
        # type: (Text, List[Event]) -> None
        """Add some more events to the tracker of a conversation."""

        url = "/conversations/{}/tracker/events".format(sender_id)

        data = [event.as_dict() for event in events]
        result = self.core_endpoint.request(subpath=url, method="post",
                                            json=data)
        result.raise_for_status()
        return result.json()

    def parse(self, message, sender_id):
        # type: (Text, Text) -> Optional[Dict[Text, Any]]
        """Send a parse request to a rasa core server."""

        url = "/conversations/{}/parse".format(sender_id)

        data = json.dumps({"query": message}, ensure_ascii=False)

        response = self.core_endpoint.request(
                subpath=url,
                method="post",
                data=data.encode("utf-8"),
                headers={'Content-Type': 'application/json; charset=utf-8'}
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.warn("Got a bad response from rasa core :( Status: {} "
                        "Response: {}".format(response.status_code,
                                              response.text))
            return None

    def upload_model(self, model_dir, max_retries=1):
        # type: (Text, int) -> Optional[Dict[Text, Any]]
        """Upload a Rasa core model to the remote instance."""

        logger.debug("Uploading model to rasa core server.")

        model_zip = utils.zip_folder(model_dir)

        response = None
        while max_retries > 0:
            max_retries -= 1

            try:
                with io.open(model_zip, "rb") as f:
                    response = self.core_endpoint.request(
                            method="post",
                            subpath="/load",
                            files={"model": f},
                            content_type=None)

                if response.status_code == 200:
                    logger.debug("Finished uploading")
                    return response.json()
            except RequestException as e:
                logger.warn("Failed to send model upload request. "
                            "{}".format(e))

            if max_retries > 0:
                # some resting time before we try again - e.g. server
                # might be unavailable / not started yet
                time.sleep(2)

        if response:
            logger.warn("Got a bad response from rasa core while uploading "
                        "the model (Status: {} "
                        "Response: {})".format(response.status_code,
                                               response.text))
        return None

    def continue_core(self, action_name, events, sender_id):
        # type: (Text, List[Event], Text) -> Optional[Dict[Text, Any]]
        """Send a continue request to rasa core to get next action
        prediction."""

        url = "/conversations/{}/continue".format(sender_id)
        dumped_events = []
        for e in events:
            dumped_events.append(e.as_dict())
        data = json.dumps(
                {"executed_action": action_name, "events": dumped_events},
                ensure_ascii=False)
        response = self.core_endpoint.request(
                method="post",
                subpath=url,
                data=data.encode('utf-8'),
                headers={'Content-type': 'text/plain; charset=utf-8'})

        if response.status_code == 200:
            return response.json()
        else:
            logger.warn("Got a bad response from rasa core :( Status: {} "
                        "Response: {}".format(response.status_code,
                                              response.text))
            return None


class RemoteAgent(object):
    """A special agent that is connected to a model running on another server.
    """

    def __init__(
            self,
            domain,  # type: Union[Text, Domain]
            core_client,  # type: RasaCoreClient
            nlg_endpoint  # type: Optional[EndpointConfig]
    ):
        self.domain = domain
        self.core_client = core_client
        self.nlg = NaturalLanguageGenerator.create(nlg_endpoint, self.domain)

    def handle_channel(
            self,
            input_channel,  # type: InputChannel
            message_preprocessor=None  # type: Optional[Callable[[Text], Text]]
    ):
        # type: (...) -> None
        """Handle messages from the input channel using remote core."""

        def message_handler(message):
            if message_preprocessor is not None:
                message.text = message_preprocessor(message.text)
            self.process_message(message)

        logger.info("Starting sync listening on input channel")
        input_channel.start_sync_listening(message_handler)

    def _run_next_action(self, action_name, message):
        # type: (Text, UserMessage) -> Dict[Text, Any]
        """Run the next action communicating with the remote core server."""

        tracker = self.core_client.tracker(message.sender_id,
                                           self.domain)
        dispatcher = Dispatcher(message.sender_id,
                                message.output_channel,
                                self.nlg)

        action = self.domain.action_for_name(action_name)
        # events and return values are used to update
        # the tracker state after an action has been taken
        try:
            action_events = action.run(dispatcher, tracker, self.domain)
        except Exception:
            logger.exception(
                    "Encountered an exception while running action "
                    "'{}'. Bot will continue, but the actions "
                    "events are lost. Make sure to fix the "
                    "exception in your custom code."
                    "".format(action.name()))
            action_events = []

        # this is similar to what is done in the processor, but instead of
        # logging the events on the tracker we need to return them to the
        # remote core instance
        events = []
        for m in dispatcher.latest_bot_messages:
            events.append(BotUttered(text=m.text, data=m.data))

        events.extend(action_events)
        return self.core_client.continue_core(action_name,
                                              events,
                                              message.sender_id)

    def process_message(self, message):
        # type: (UserMessage) -> None
        """Process a message using a remote rasa core instance."""

        try:
            response = self.core_client.parse(message.text, message.sender_id)

            while (response and
                   response.get("next_action") != ACTION_LISTEN_NAME):

                action_name = response.get("next_action")
                if action_name is not None:
                    response = self._run_next_action(action_name, message)
                else:
                    logger.error("Rasa Core did not return an action. "
                                 "Response: {}".format(response))
                    break

        except Exception:
            logger.exception("Failed to process message.")
        else:
            logger.info("Done processing message")

    @classmethod
    def load(cls,
             path,  # type: Text
             core_endpoint,  # type: EndpointConfig
             nlg_endpoint=None,  # type: EndpointConfig
             action_factory=None  # type: Optional[Text]
             ):
        # type: (...) -> RemoteAgent

        if isinstance(core_endpoint, string_types):
            raise Exception("This API has changed. Instead of passing in a url "
                            "for Rasa Core, you now need to pass in an "
                            "instance of 'EndpointConfig'. "
                            "(from rasa_core.utils import EndpointConfig )")

        domain = TemplateDomain.load(os.path.join(path, "domain.yml"),
                                     action_factory)

        core_client = RasaCoreClient(core_endpoint)
        core_client.upload_model(path, max_retries=5)

        return RemoteAgent(domain, core_client, nlg_endpoint)
