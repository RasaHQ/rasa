from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import typing

import json
import logging

# noinspection PyPep8Naming
import six.moves.cPickle as pickler
from typing import Text, Optional, List

from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.broker import EventChannel
from rasa_core.trackers import (
    DialogueStateTracker, ActionExecuted,
    EventVerbosity)

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain


class TrackerStore(object):
    def __init__(self, domain, event_broker=None):
        # type: (Optional[Domain], Optional[EventChannel]) -> None
        self.domain = domain
        self.event_broker = event_broker

    def get_or_create_tracker(self, sender_id):
        tracker = self.retrieve(sender_id)
        if tracker is None:
            tracker = self.create_tracker(sender_id)
        return tracker

    def init_tracker(self, sender_id):
        if self.domain:
            return DialogueStateTracker(sender_id,
                                        self.domain.slots)
        else:
            return None

    def create_tracker(self, sender_id, append_action_listen=True):
        """Creates a new tracker for the sender_id.

        The tracker is initially listening."""

        tracker = self.init_tracker(sender_id)
        if tracker:
            if append_action_listen:
                tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
            self.save(tracker)
        return tracker

    def save(self, tracker):
        raise NotImplementedError()

    def retrieve(self, sender_id):
        # type: (Text) -> Optional[DialogueStateTracker]
        raise NotImplementedError()

    def stream_events(self, tracker):
        # type: (DialogueStateTracker) -> None
        old_tracker = self.retrieve(tracker.sender_id)
        offset = len(old_tracker.events) if old_tracker else 0
        evts = tracker.events
        for evt in list(itertools.islice(evts, offset, len(evts))):
            body = {
                "sender_id": tracker.sender_id,
            }
            body.update(evt.as_dict())
            self.event_broker.publish(json.dumps(body))

    def keys(self):
        # type: () -> Optional[List[Text]]
        raise NotImplementedError()

    @staticmethod
    def serialise_tracker(tracker):
        dialogue = tracker.as_dialogue()
        return pickler.dumps(dialogue)

    def deserialise_tracker(self, sender_id, _json):
        dialogue = pickler.loads(_json)
        tracker = self.init_tracker(sender_id)
        tracker.recreate_from_dialogue(dialogue)
        return tracker


class InMemoryTrackerStore(TrackerStore):
    def __init__(self, domain, event_broker=None):
        self.store = {}
        super(InMemoryTrackerStore, self).__init__(domain, event_broker)

    def save(self, tracker):
        if self.event_broker:
            self.stream_events(tracker)
        serialised = InMemoryTrackerStore.serialise_tracker(tracker)
        self.store[tracker.sender_id] = serialised

    def retrieve(self, sender_id):
        if sender_id in self.store:
            logger.debug('Recreating tracker for '
                         'id \'{}\''.format(sender_id))
            return self.deserialise_tracker(sender_id, self.store[sender_id])
        else:
            logger.debug('Creating a new tracker for '
                         'id \'{}\'.'.format(sender_id))
            return None

    def keys(self):
        return self.store.keys()


class RedisTrackerStore(TrackerStore):
    def keys(self):
        pass

    def __init__(self, domain, host='localhost',
                 port=6379, db=0, password=None, event_broker=None,
                 record_exp=None):

        import redis
        self.red = redis.StrictRedis(host=host, port=port, db=db,
                                     password=password)
        self.record_exp = record_exp
        super(RedisTrackerStore, self).__init__(domain, event_broker)

    def save(self, tracker, timeout=None):
        if self.event_broker:
            self.stream_events(tracker)

        if not timeout and self.record_exp:
            timeout = self.record_exp

        serialised_tracker = self.serialise_tracker(tracker)
        self.red.set(tracker.sender_id, serialised_tracker, ex=timeout)

    def retrieve(self, sender_id):
        stored = self.red.get(sender_id)
        if stored is not None:
            return self.deserialise_tracker(sender_id, stored)
        else:
            return None


class MongoTrackerStore(TrackerStore):
    def __init__(self,
                 domain,
                 host="mongodb://localhost:27017",
                 db="rasa",
                 username=None,
                 password=None,
                 collection="conversations",
                 event_broker=None):
        from pymongo.database import Database
        from pymongo import MongoClient

        self.client = MongoClient(host,
                                  username=username,
                                  password=password,
                                  # delay connect until process forking is done
                                  connect=False)

        self.db = Database(self.client, db)
        self.collection = collection
        super(MongoTrackerStore, self).__init__(domain, event_broker)

        self._ensure_indices()

    @property
    def conversations(self):
        return self.db[self.collection]

    def _ensure_indices(self):
        self.conversations.create_index("sender_id")

    def save(self, tracker, timeout=None):
        if self.event_broker:
            self.stream_events(tracker)

        state = tracker.current_state(EventVerbosity.ALL)

        self.conversations.update_one(
                {"sender_id": tracker.sender_id},
                {"$set": state},
                upsert=True)

    def retrieve(self, sender_id):
        stored = self.conversations.find_one({"sender_id": sender_id})

        # look for conversations which have used an `int` sender_id in the past
        # and update them.
        if stored is None and sender_id.isdigit():
            from pymongo import ReturnDocument
            stored = self.conversations.find_one_and_update(
                {"sender_id": int(sender_id)},
                {"$set": {"sender_id": str(sender_id)}},
                return_document=ReturnDocument.AFTER)

        if stored is not None:
            if self.domain:
                return DialogueStateTracker.from_dict(sender_id,
                                                      stored.get("events"),
                                                      self.domain.slots)
            else:
                logger.warning("Can't recreate tracker from mongo storage "
                               "because no domain is set. Returning `None` "
                               "instead.")
                return None
        else:
            return None

    def keys(self):
        return [c["sender_id"] for c in self.conversations.find()]
