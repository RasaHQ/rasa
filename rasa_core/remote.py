from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time

import io
import json
import logging
from requests.exceptions import RequestException
from typing import Text, Optional, Dict, Any

from rasa_core import utils
from rasa_core.domain import Domain
from rasa_core.events import Event
from rasa_core.trackers import DialogueStateTracker, EventVerbosity
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
                event_verbosity=EventVerbosity.ALL,  # type: EventVerbosity
                until=None  # type: Optional[int]
                ):
        """Retrieve and recreate a tracker fetched from the remote instance."""

        tracker_json = self.tracker_json(
                sender_id, event_verbosity, until)

        tracker = DialogueStateTracker.from_dict(
                sender_id, tracker_json.get("events", []), domain.slots)
        return tracker

    def tracker_json(self,
                     sender_id,  # type: Text
                     event_verbosity=EventVerbosity.ALL,  # type: EventVerbosity
                     until=None  # type: Optional[int]
                     ):
        """Retrieve a tracker's json representation from remote instance."""

        url = "/conversations/{}/tracker?include_events={}".format(
                sender_id, event_verbosity.name)
        if until:
            url += "&until={}".format(until)

        result = self.core_endpoint.request(subpath=url,
                                            method="get")
        result.raise_for_status()
        return result.json()

    def append_event_to_tracker(self, sender_id, event):
        # type: (Text, Event) -> None
        """Add some more events to the tracker of a conversation."""

        url = "/conversations/{}/tracker/events".format(sender_id)

        result = self.core_endpoint.request(subpath=url, method="post",
                                            json=event.as_dict())
        result.raise_for_status()
        return result.json()

    def respond(self, message, sender_id):
        # type: (Text, Text) -> Optional[Dict[Text, Any]]
        """Send a parse request to a rasa core server."""

        url = "/conversations/{}/respond".format(sender_id)

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
            logger.warning("Got a bad response from rasa core :( Status: {} "
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
                logger.warning("Failed to send model upload request. "
                               "{}".format(e))

            if max_retries > 0:
                # some resting time before we try again - e.g. server
                # might be unavailable / not started yet
                time.sleep(10)

        if response:
            logger.warning("Got a bad response from rasa core while uploading "
                           "the model (Status: {} "
                           "Response: {})".format(response.status_code,
                                                  response.text))
        return None
