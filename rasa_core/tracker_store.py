import itertools

import json
import time
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
from sqlalchemy import Table, Column, Integer, String, Float, MetaData

logger = logging.getLogger(__name__)


class TrackerStore(object):
    def __init__(self,
                 domain: Optional[Domain],
                 event_broker: Optional[EventChannel] = None) -> None:
        self.domain = domain
        self.event_broker = event_broker
        self.max_event_history = None

    @staticmethod
    def find_tracker_store(domain, store=None, event_broker=None):
        if store is None or store.type is None:
            return InMemoryTrackerStore(domain, event_broker=event_broker)
        elif store.type == 'redis':
            return RedisTrackerStore(domain=domain,
                                     host=store.url,
                                     event_broker=event_broker,
                                     **store.kwargs)
        elif store.type == 'mongod':
            return MongoTrackerStore(domain=domain,
                                     host=store.url,
                                     event_broker=event_broker,
                                     **store.kwargs)
        elif store.type == 'SQL':
            return SQLTrackerStore(domain=domain,
                                   host=store.url,
                                   event_broker=event_broker,
                                   **store.kwargs)
        else:
            return TrackerStore.load_tracker_from_module_string(domain, store)

    @staticmethod
    def load_tracker_from_module_string(domain, store):
        custom_tracker = None
        try:
            custom_tracker = class_from_module_path(store.type)
        except (AttributeError, ImportError):
            logger.warning("Store type '{}' not found. "
                           "Using InMemoryTrackerStore instead"
                           .format(store.type))

        if custom_tracker:
            return custom_tracker(domain=domain,
                                  url=store.url, **store.kwargs)
        else:
            return InMemoryTrackerStore(domain)

    def get_or_create_tracker(self, sender_id, max_event_history=None):
        tracker = self.retrieve(sender_id)
        self.max_event_history = max_event_history
        if tracker is None:
            tracker = self.create_tracker(sender_id)
        return tracker

    def init_tracker(self, sender_id):
        if self.domain:
            return DialogueStateTracker(
                sender_id,
                self.domain.slots,
                max_event_history=self.max_event_history)
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
            self.event_broker.publish(body)

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


class SQLTrackerStore(TrackerStore):
    """Store which can save and retrieve trackers from an SQL database"""

    from sqlalchemy.ext.declarative import declarative_base
    Base = declarative_base()

    class SQLEvent(Base):
        __tablename__ = 'conversations'

        id = Column(Integer, primary_key=True)
        sender_id = Column(String, nullable=False)
        type_name = Column(String, nullable=False)
        timestamp = Column(Float)
        intent_name = Column(String)
        action_name = Column(String)
        data = Column(String)

    def __init__(self,
                 domain: Optional[Domain] = None,
                 drivername: Text = 'sqlite',
                 host: Text = None,
                 event_broker: Optional[EventChannel] = None,
                 db: Text = '',
                 username: Text = None,
                 password: Text = None) -> None:
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.engine.url import URL
        from sqlalchemy import create_engine

        engine_url = URL(drivername, username, password, host, database=db)

        logger.debug('Attempting to connect to database '
                     'via "{}"...'.format(engine_url.__to_string__()))

        self.engine = create_engine(engine_url)
        self.session = sessionmaker(bind=self.engine)()

        logger.debug('Connection successful, ensuring conversations table...')

        self._ensure_table()
        super(SQLTrackerStore, self).__init__(domain, event_broker)

        logger.debug('SQL tracker store successfully initialised')

    def _ensure_table(self):
        """Creates the conversations table in the database if not present"""

        metadata = MetaData()

        Table("conversations", metadata,
              Column("id", Integer, primary_key=True),
              Column("sender_id", String, nullable=False),
              Column("type_name", String, nullable=False),
              Column("timestamp", Float),
              Column("intent_name", String),
              Column("action_name", String),
              Column("data", String))

        metadata.create_all(self.engine)

    def keys(self):
        """Returns the keys of the items stored in the database"""
        return self.SQLEvent.__table__.columns.keys()

    def retrieve(self, sender_id: Text):
        """Recreates the tracker from all previously stored events"""

        subquery = self.session.query(self.SQLEvent)
        query = subquery.filter_by(sender_id=sender_id).all()
        events = [json.loads(event.data) for event in query]

        if self.domain and len(events) > 0:
            logger.debug('Recreating tracker '
                         'from {} stored events'.format(len(events)))

            tracker = DialogueStateTracker.from_dict(sender_id, events,
                                                     self.domain.slots)
        else:
            logger.warning("Can't retrieve tracker from SQL storage.  "
                           "Returning `None` instead.")
            tracker = None
        return tracker

    def save(self, tracker):
        """Updates database with events from the current conversation"""

        if self.event_broker:
            self.stream_events(tracker)

        events = self._event_buffer(tracker)  # only store recent events

        for event in events:
            try:
                data = event.as_dict()
            except AttributeError as e:
                logger.warning("Unable to serialise event: {}.  Using "
                               "__dict__ method instead".format(e))
                data = event.__dict__

            intent = data.get("parse_data", {}).get("intent", {}).get("name")
            action = data.get("name")
            timestamp = data.get("timestamp")

            self.session.add(self.SQLEvent(sender_id=tracker.sender_id,
                                           type_name=event.type_name,
                                           timestamp=timestamp,
                                           intent_name=intent,
                                           action_name=action,
                                           data=json.dumps(data)))
        self.session.commit()
        logger.debug('Tracker stored to database')

    def _event_buffer(self, tracker):
        """Returns events from the tracker which aren't currently stored"""

        from sqlalchemy import func
        query = self.session.query(func.max(self.SQLEvent.timestamp))
        max_timestamp = query.filter_by(sender_id=tracker.sender_id).scalar()

        if max_timestamp is None:
            max_timestamp = 0

        latest_events = []

        for event in reversed(tracker.events):
            if event.timestamp > max_timestamp:
                latest_events.append(event)
            else:
                break

        logger.debug('Storing {} recent events'.format(len(latest_events)))
        return reversed(latest_events)
