from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import logging
import time

import requests
from future.moves.urllib.parse import quote_plus
from requests.exceptions import RequestException
from typing import Text, List, Optional, Dict, Any

from rasa_core import utils
from rasa_core.domain import Domain
from rasa_core.events import Event
from rasa_core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


class RasaCoreClient(object):
    """Connects to a running Rasa Core server.

    Used to retrieve information about models and conversations."""

    def __init__(self, host="127.0.0.1:5005", token=None):
        # type: (Text, Optional[Text]) -> None

        self.host = host
        self.token = token if token else ""

    def status(self):
        """Get the status of the remote core server (e.g. the version.)"""

        url = "{}/version?token={}".format(self.host, self.token)
        result = requests.get(url)
        result.raise_for_status()
        return result.json()

    def clients(self):
        """Get a list of all conversations."""

        url = "{}/conversations?token={}".format(self.host, self.token)
        result = requests.get(url)
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
                sender_id, tracker_json.get("events", []), domain.slots)
        return tracker

    def tracker_json(self,
                     sender_id,  # type: Text
                     use_history=True,  # type: bool
                     include_events=True,  # type: bool
                     until=None  # type: Optional[int]
                     ):
        """Retrieve a tracker's json representation from remote instance."""

        url = ("{}/conversations/{}/tracker?token={}"
               "&ignore_restarts={}"
               "&events={}").format(self.host, sender_id, self.token,
                                    use_history, include_events)
        if until:
            url += "&until={}".format(until)

        result = requests.get(url)
        result.raise_for_status()
        return result.json()

    def append_events_to_tracker(self, sender_id, events):
        # type: (Text, List[Event]) -> None
        """Add some more events to the tracker of a conversation."""

        url = "{}/conversations/{}/tracker/events?token={}".format(
                self.host, sender_id, self.token)

        data = [event.as_dict() for event in events]
        result = requests.post(url, json=data)
        result.raise_for_status()
        return result.json()

    def respond(self, message, sender_id):
        # type: (Text, Text) -> Optional[Dict[Text, Any]]
        """Send a parse request to a rasa core server."""

        url = "{}/conversations/{}/respond?token={}".format(
                self.host, sender_id, quote_plus(self.token))

        data = json.dumps({"query": message}, ensure_ascii=False)

        response = requests.post(url, data=data.encode("utf-8"),
                                 headers={
                                     'Content-type': 'text/plain; '
                                                     'charset=utf-8'})

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

        url = "{}/load?token={}".format(self.host, quote_plus(self.token))
        logger.debug("Uploading model to rasa core server.")

        model_zip = utils.zip_folder(model_dir)

        response = None
        while max_retries > 0:
            max_retries -= 1

            try:
                with io.open(model_zip, "rb") as f:
                    response = requests.post(url, files={"model": f})

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


