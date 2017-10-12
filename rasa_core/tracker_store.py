from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import logging

import six.moves.cPickle as pickler

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

    def _init_tracker(self, sender_id):
        return DialogueStateTracker(sender_id,
                                       self.domain.slots,
                                       self.domain.topics,
                                       self.domain.default_topic)

    def create_tracker(self, sender_id):
        """Creates a new tracker for the sender.

        The tracker is initially listening."""

        tracker = self._init_tracker(sender_id)
        tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
        self.save(tracker)
        return tracker

    def save(self, tracker):
        raise NotImplementedError()

    def retrieve(self, sender_id):
        raise NotImplementedError()

    @staticmethod
    def serialise_tracker(tracker):
        dialogue = tracker.as_dialogue()
        return pickler.dumps(dialogue)

    def deserialise_tracker(self, sender_id, _json):
        dialogue = pickler.loads(_json)
        tracker = self._init_tracker(sender_id)
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
            logger.debug('Could not find a tracker for '
                         'id \'{}\''.format(sender_id))
            return None


class RedisTrackerStore(TrackerStore):

    def __init__(self, domain, mock=False, host='localhost',
                 port=6379, db=0, password=None):

        if mock:
            import fakeredis
            self.red = fakeredis.FakeStrictRedis()
        else:  # pragma: no cover
            import redis
            self.red = redis.StrictRedis(host=host, port=port, db=db,
                                         password=password)
        super(RedisTrackerStore, self).__init__(domain)

    def save(self, tracker, timeout=None):
        serialised_tracker = RedisTrackerStore.serialise_tracker(tracker)
        self.red.set(tracker.sender_id, serialised_tracker, ex=timeout)

    def retrieve(self, sender_id):
        stored = self.red.get(sender_id)
        if stored is not None:
            return self.deserialise_tracker(sender_id, stored)
        else:
            return None
