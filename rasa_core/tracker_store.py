import itertools

import json
import logging
import pickle
# noinspection PyPep8Naming
from typing import Text, Optional, List, KeysView

from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.broker import EventChannel
from rasa_core.domain import Domain
from rasa_core.trackers import (
    DialogueStateTracker, ActionExecuted,
    EventVerbosity)
from rasa_core.utils import class_from_module_path

logger = logging.getLogger(__name__)


class TrackerStore(object):
    def __init__(self,
                 domain: Optional[Domain],
                 event_broker: Optional[EventChannel] = None) -> None:
        self.domain = domain
        self.event_broker = event_broker

    @staticmethod
    def find_tracker_store(domain, store=None, event_broker=None):
        if store is None or store.store_type is None:
            return InMemoryTrackerStore(domain, event_broker=event_broker)
        elif store.store_type == 'redis':
            return RedisTrackerStore(domain=domain,
                                     host=store.url,
                                     event_broker=event_broker,
                                     **store.kwargs)
        elif store.store_type == 'mongod':
            return MongoTrackerStore(domain=domain,
                                     host=store.url,
                                     event_broker=event_broker,
                                     **store.kwargs)
        else:
            return TrackerStore.load_tracker_from_module_string(domain, store)

    @staticmethod
    def load_tracker_from_module_string(domain, store):
        custom_tracker = None
        try:
            custom_tracker = class_from_module_path(store.store_type)
        except (AttributeError, ImportError):
            logger.warning("Store type {} not found. "
                           "Using InMemoryTrackerStore instead"
                           .format(store.store_type))

        if custom_tracker:
            return custom_tracker(domain=domain,
                                  url=store.url, **store.kwargs)
        else:
            return InMemoryTrackerStore(domain)

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

    def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        raise NotImplementedError()

    def stream_events(self, tracker: DialogueStateTracker) -> None:
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
        return pickle.dumps(dialogue)

    def deserialise_tracker(self, sender_id, _json):
        dialogue = pickle.loads(_json)
        tracker = self.init_tracker(sender_id)
        tracker.recreate_from_dialogue(dialogue)
        return tracker


class InMemoryTrackerStore(TrackerStore):
    def __init__(self,
                 domain: Domain,
                 event_broker: Optional[EventChannel] = None
                 ) -> None:
        self.store = {}
        super(InMemoryTrackerStore, self).__init__(domain, event_broker)

    def save(self, tracker: DialogueStateTracker) -> None:
        if self.event_broker:
            self.stream_events(tracker)
        serialised = InMemoryTrackerStore.serialise_tracker(tracker)
        self.store[tracker.sender_id] = serialised

    def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        if sender_id in self.store:
            logger.debug('Recreating tracker for '
                         'id \'{}\''.format(sender_id))
            return self.deserialise_tracker(sender_id, self.store[sender_id])
        else:
            logger.debug('Creating a new tracker for '
                         'id \'{}\'.'.format(sender_id))
            return None

    def keys(self) -> KeysView[Text]:
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
                 auth_source="admin",
                 collection="conversations",
                 event_broker=None):
        from pymongo.database import Database
        from pymongo import MongoClient

        self.client = MongoClient(host,
                                  username=username,
                                  password=password,
                                  authSource=auth_source,
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


from sqlalchemy import Table, Column, Integer, String, Float, Unicode


class SQLTrackerStore(TrackerStore):
    """Store which can save and retrieve trackers from an SQL database"""

    from sqlalchemy.ext.declarative import declarative_base
    Base = declarative_base()

    class SQLEvent(Base):
        __tablename__ = 'events'

        id = Column(Integer, primary_key=True)
        sender_id = Column(String, nullable=False)
        type_name = Column(String, nullable=False)
        timestamp = Column(Float, nullable=False)
        intent = Column(String)
        action = Column(String)
        data = Column(String)

    def __init__(self,
                 domain: Optional[Domain],
                 drivername: Text = 'sqlite',
                 host: Text = None,
                 port: int = None,
                 db: Text = 'rasa',
                 username: Text = None,
                 password: Text = None,
                 event_broker: Optional[EventChannel] = None) -> None:
        from sqlalchemy import MetaData, create_engine
        from sqlalchemy.engine.url import URL
        from sqlalchemy.orm import sessionmaker

        engine_url = URL(drivername, username, password, host, port, db)

        self.engine = create_engine(engine_url)
        self.Session = sessionmaker(bind=self.engine)
        self.conn = self.engine.connect()
        self.metadata = MetaData()
        self.domain = domain
        self.event_broker = event_broker
        super(SQLTrackerStore, self).__init__(domain, event_broker)

        self.session = self.Session()
        self.ensure_event_table()

    def ensure_event_table(self):
        """Creates the events table if not already present in the database"""
        Table("events", self.metadata,
              Column("id", Integer, primary_key=True),
              Column("sender_id", String, nullable=False),
              Column("type_name", String, nullable=False),
              Column("timestamp", Float, nullable=False),
              Column("intent", String),
              Column("action", String),
              Column("data", String))

        self.metadata.create_all(self.engine)

    def keys(self):
        pass

    def retrieve(self, sender_id: Text):
        """Recreates the tracker from all previously stored events"""

        import ast

        query = self.session.query(self.SQLEvent).filter_by(sender_id=sender_id).all()
        events = [ast.literal_eval(event.data) for event in query]

        if self.domain:
            return DialogueStateTracker.from_dict(sender_id,
                                                  events,
                                                  self.domain.slots)
        else:
            logger.warning("Can't recreate tracker from SQL storage "
                           "because no domain is set. Returning `None` "
                           "instead.")
            return None

    def save(self, tracker):
        """Updates database with events from the current conversation"""

        if self.event_broker:
            self.stream_events(tracker)

        for event in tracker.events:
            event_data = event.as_dict()
            intent = event_data.get("parse_data", {}).get("intent")
            action = event_data.get("name")  # works for reminder, slotset, form, followupactions...

            self.session.add(self.SQLEvent(sender_id=tracker.sender_id,
                                           type_name=event.type_name,
                                           timestamp=event.timestamp,
                                           intent=intent,
                                           action=action,
                                           data=str(event_data)))
        self.session.commit()
