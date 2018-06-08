from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import logging

import six.moves.cPickle as pickler
from typing import Text, Optional

from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.trackers import DialogueStateTracker, ActionExecuted

logger = logging.getLogger(__name__)


class TrackerStore(object):
    def __init__(self, domain):
        self.domain = domain

    def get_or_create_tracker(self, sender_id):
        tracker = self.retrieve(sender_id)
        if tracker is None:
            tracker = self.create_tracker(sender_id)
        return tracker

    def init_tracker(self, sender_id):
        return DialogueStateTracker(sender_id,
                                    self.domain.slots)

    def create_tracker(self, sender_id, append_action_listen=True):
        """Creates a new tracker for the sender_id.

        The tracker is initially listening."""

        tracker = self.init_tracker(sender_id)
        if append_action_listen:
            tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
        self.save(tracker)
        return tracker

    def save(self, tracker):
        raise NotImplementedError()

    def retrieve(self, sender_id):
        # type: (Text) -> Optional[DialogueStateTracker]
        raise NotImplementedError()

    def keys(self):
        # type: (Text) -> List[Text]
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
    def __init__(self, domain):

        self.store = {}
        super(InMemoryTrackerStore, self).__init__(domain)

    def save(self, tracker):
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
                 port=6379, db=0, password=None):

        import redis
        self.red = redis.StrictRedis(host=host, port=port, db=db,
                                     password=password)
        super(RedisTrackerStore, self).__init__(domain)

    def save(self, tracker, timeout=None):
        serialised_tracker = self.serialise_tracker(tracker)
        self.red.set(tracker.sender_id, serialised_tracker, ex=timeout)

    def retrieve(self, sender_id):
        stored = self.red.get(sender_id)
        if stored is not None:
            return self.deserialise_tracker(sender_id, stored)
        else:
            return None
