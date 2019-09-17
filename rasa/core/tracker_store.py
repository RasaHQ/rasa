import contextlib
import json
import logging
import os
import pickle
import typing
from typing import Iterator, Optional, Text, Iterable, Union, Dict

import itertools

# noinspection PyPep8Naming
from time import sleep

from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.brokers.event_channel import EventChannel
from rasa.core.domain import Domain
from rasa.core.trackers import ActionExecuted, DialogueStateTracker, EventVerbosity
from rasa.utils.common import class_from_module_path

if typing.TYPE_CHECKING:
    from sqlalchemy.engine.url import URL
    from sqlalchemy.engine.base import Engine
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class TrackerStore(object):
    def __init__(
        self, domain: Optional[Domain], event_broker: Optional[EventChannel] = None
    ) -> None:
        self.domain = domain
        self.event_broker = event_broker
        self.max_event_history = None

    @staticmethod
    def find_tracker_store(domain, store=None, event_broker=None):
        if store is None or store.type is None:
            tracker_store = InMemoryTrackerStore(domain, event_broker=event_broker)
        elif store.type == "redis":
            tracker_store = RedisTrackerStore(
                domain=domain, host=store.url, event_broker=event_broker, **store.kwargs
            )
        elif store.type == "mongod":
            tracker_store = MongoTrackerStore(
                domain=domain, host=store.url, event_broker=event_broker, **store.kwargs
            )
        elif store.type.lower() == "sql":
            tracker_store = SQLTrackerStore(
                domain=domain, host=store.url, event_broker=event_broker, **store.kwargs
            )
        else:
            tracker_store = TrackerStore.load_tracker_from_module_string(domain, store)

        logger.debug("Connected to {}.".format(tracker_store.__class__.__name__))
        return tracker_store

    @staticmethod
    def load_tracker_from_module_string(domain, store):
        custom_tracker = None
        try:
            custom_tracker = class_from_module_path(store.type)
        except (AttributeError, ImportError):
            logger.warning(
                "Store type '{}' not found. "
                "Using InMemoryTrackerStore instead".format(store.type)
            )

        if custom_tracker:
            return custom_tracker(domain=domain, url=store.url, **store.kwargs)
        else:
            return InMemoryTrackerStore(domain)

    def get_or_create_tracker(self, sender_id, max_event_history=None):
        tracker = self.retrieve(sender_id)
        self.max_event_history = max_event_history
        if tracker is None:
            tracker = self.create_tracker(sender_id)
        return tracker

    def init_tracker(self, sender_id):
        return DialogueStateTracker(
            sender_id,
            self.domain.slots if self.domain else None,
            max_event_history=self.max_event_history,
        )

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
        offset = self.number_of_existing_events(tracker.sender_id)
        evts = tracker.events
        for evt in list(itertools.islice(evts, offset, len(evts))):
            body = {"sender_id": tracker.sender_id}
            body.update(evt.as_dict())
            self.event_broker.publish(body)

    def number_of_existing_events(self, sender_id: Text) -> int:
        """Return number of stored events for a given sender id."""
        old_tracker = self.retrieve(sender_id)
        return len(old_tracker.events) if old_tracker else 0

    def keys(self) -> Iterable[Text]:
        raise NotImplementedError()

    @staticmethod
    def serialise_tracker(tracker):
        dialogue = tracker.as_dialogue()
        return pickle.dumps(dialogue)

    def deserialise_tracker(self, sender_id, _json) -> Optional[DialogueStateTracker]:
        dialogue = pickle.loads(_json)
        tracker = self.init_tracker(sender_id)
        if tracker:
            tracker.recreate_from_dialogue(dialogue)
            return tracker
        else:
            return None


class InMemoryTrackerStore(TrackerStore):
    def __init__(
        self, domain: Domain, event_broker: Optional[EventChannel] = None
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
            logger.debug("Recreating tracker for id '{}'".format(sender_id))
            return self.deserialise_tracker(sender_id, self.store[sender_id])
        else:
            logger.debug("Creating a new tracker for id '{}'.".format(sender_id))
            return None

    def keys(self) -> Iterable[Text]:
        return self.store.keys()


class RedisTrackerStore(TrackerStore):
    def keys(self) -> Iterable[Text]:
        return self.red.keys()

    def __init__(
        self,
        domain,
        host="localhost",
        port=6379,
        db=0,
        password=None,
        event_broker=None,
        record_exp=None,
    ):

        import redis

        self.red = redis.StrictRedis(host=host, port=port, db=db, password=password)
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
    def __init__(
        self,
        domain,
        host="mongodb://localhost:27017",
        db="rasa",
        username=None,
        password=None,
        auth_source="admin",
        collection="conversations",
        event_broker=None,
    ):
        from pymongo.database import Database
        from pymongo import MongoClient

        self.client = MongoClient(
            host,
            username=username,
            password=password,
            authSource=auth_source,
            # delay connect until process forking is done
            connect=False,
        )

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
            {"sender_id": tracker.sender_id}, {"$set": state}, upsert=True
        )

    def retrieve(self, sender_id):
        stored = self.conversations.find_one({"sender_id": sender_id})

        # look for conversations which have used an `int` sender_id in the past
        # and update them.
        if stored is None and sender_id.isdigit():
            from pymongo import ReturnDocument

            stored = self.conversations.find_one_and_update(
                {"sender_id": int(sender_id)},
                {"$set": {"sender_id": str(sender_id)}},
                return_document=ReturnDocument.AFTER,
            )

        if stored is not None:
            if self.domain:
                return DialogueStateTracker.from_dict(
                    sender_id, stored.get("events"), self.domain.slots
                )
            else:
                logger.warning(
                    "Can't recreate tracker from mongo storage "
                    "because no domain is set. Returning `None` "
                    "instead."
                )
                return None
        else:
            return None

    def keys(self) -> Iterable[Text]:
        return [c["sender_id"] for c in self.conversations.find()]


class SQLTrackerStore(TrackerStore):
    """Store which can save and retrieve trackers from an SQL database."""

    from sqlalchemy.ext.declarative import declarative_base

    Base = declarative_base()

    class SQLEvent(Base):
        from sqlalchemy import Column, Integer, String, Float, Text

        __tablename__ = "events"

        id = Column(Integer, primary_key=True)
        sender_id = Column(String(255), nullable=False, index=True)
        type_name = Column(String(255), nullable=False)
        timestamp = Column(Float)
        intent_name = Column(String(255))
        action_name = Column(String(255))
        data = Column(Text)

    def __init__(
        self,
        domain: Optional[Domain] = None,
        dialect: Text = "sqlite",
        host: Optional[Text] = None,
        port: Optional[int] = None,
        db: Text = "rasa.db",
        username: Text = None,
        password: Text = None,
        event_broker: Optional[EventChannel] = None,
        login_db: Optional[Text] = None,
        query: Optional[Dict] = None,
    ) -> None:
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import create_engine
        import sqlalchemy.exc

        engine_url = self.get_db_url(
            dialect, host, port, db, username, password, login_db, query
        )
        logger.debug(
            "Attempting to connect to database via '{}'.".format(repr(engine_url))
        )

        # Database might take a while to come up
        while True:
            try:
                # pool_size and max_overflow can be set to control the number of
                # connections that are kept in the connection pool. Not available
                # for SQLite, and only  tested for postgresql. See
                # https://docs.sqlalchemy.org/en/13/core/pooling.html#sqlalchemy.pool.QueuePool
                if dialect == "postgresql":
                    self.engine = create_engine(
                        engine_url,
                        pool_size=int(os.environ.get("SQL_POOL_SIZE", "50")),
                        max_overflow=int(os.environ.get("SQL_MAX_OVERFLOW", "100")),
                    )
                else:
                    self.engine = create_engine(engine_url)

                # if `login_db` has been provided, use current channel with
                # that database to create working database `db`
                if login_db:
                    self._create_database_and_update_engine(db, engine_url)

                try:
                    self.Base.metadata.create_all(self.engine)
                except (
                    sqlalchemy.exc.OperationalError,
                    sqlalchemy.exc.ProgrammingError,
                ) as e:
                    # Several Rasa services started in parallel may attempt to
                    # create tables at the same time. That is okay so long as
                    # the first services finishes the table creation.
                    logger.error("Could not create tables: {}".format(e))

                self.sessionmaker = sessionmaker(bind=self.engine)
                break
            except (
                sqlalchemy.exc.OperationalError,
                sqlalchemy.exc.IntegrityError,
            ) as e:

                logger.warning(e)
                sleep(5)

        logger.debug("Connection to SQL database '{}' successful.".format(db))

        super(SQLTrackerStore, self).__init__(domain, event_broker)

    @staticmethod
    def get_db_url(
        dialect: Text = "sqlite",
        host: Optional[Text] = None,
        port: Optional[int] = None,
        db: Text = "rasa.db",
        username: Text = None,
        password: Text = None,
        login_db: Optional[Text] = None,
        query: Optional[Dict] = None,
    ) -> Union[Text, "URL"]:
        """Builds an SQLAlchemy `URL` object representing the parameters needed
        to connect to an SQL database.

        Args:
            dialect: SQL database type.
            host: Database network host.
            port: Database network port.
            db: Database name.
            username: User name to use when connecting to the database.
            password: Password for database user.
            login_db: Alternative database name to which initially connect, and create
                the database specified by `db` (PostgreSQL only).
            query: Dictionary of options to be passed to the dialect and/or the
                DBAPI upon connect.

        Returns:
            URL ready to be used with an SQLAlchemy `Engine` object.

        """
        from urllib.parse import urlsplit
        from sqlalchemy.engine.url import URL

        # Users might specify a url in the host
        parsed = urlsplit(host or "")
        if parsed.scheme:
            return host

        if host:
            # add fake scheme to properly parse components
            parsed = urlsplit("schema://" + host)

            # users might include the port in the url
            port = parsed.port or port
            host = parsed.hostname or host

        return URL(
            dialect,
            username,
            password,
            host,
            port,
            database=login_db if login_db else db,
            query=query,
        )

    def _create_database_and_update_engine(self, db: Text, engine_url: "URL"):
        """Create databse `db` and update engine to reflect the updated
            `engine_url`."""

        from sqlalchemy import create_engine

        self._create_database(self.engine, db)
        engine_url.database = db
        self.engine = create_engine(engine_url)

    @staticmethod
    def _create_database(engine: "Engine", db: Text):
        """Create database `db` on `engine` if it does not exist."""

        import psycopg2

        conn = engine.connect()

        cursor = conn.connection.cursor()
        cursor.execute("COMMIT")
        cursor.execute(
            ("SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{}'".format(db))
        )
        exists = cursor.fetchone()
        if not exists:
            try:
                cursor.execute("CREATE DATABASE {}".format(db))
            except psycopg2.IntegrityError as e:
                logger.error("Could not create database '{}': {}".format(db, e))

        cursor.close()
        conn.close()

    @contextlib.contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.sessionmaker()
        try:
            yield session
        finally:
            session.close()

    def keys(self) -> Iterable[Text]:
        with self.session_scope() as session:
            sender_ids = session.query(self.SQLEvent.sender_id).distinct().all()
            return [sender_id for (sender_id,) in sender_ids]

    def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Create a tracker from all previously stored events."""

        with self.session_scope() as session:
            query = session.query(self.SQLEvent)
            result = (
                query.filter_by(sender_id=sender_id)
                .order_by(self.SQLEvent.timestamp)
                .all()
            )

            events = [json.loads(event.data) for event in result]

            if self.domain and len(events) > 0:
                logger.debug("Recreating tracker from sender id '{}'".format(sender_id))
                return DialogueStateTracker.from_dict(
                    sender_id, events, self.domain.slots
                )
            else:
                logger.debug(
                    "Can't retrieve tracker matching "
                    "sender id '{}' from SQL storage. "
                    "Returning `None` instead.".format(sender_id)
                )
                return None

    def save(self, tracker: DialogueStateTracker) -> None:
        """Update database with events from the current conversation."""

        if self.event_broker:
            self.stream_events(tracker)

        with self.session_scope() as session:
            # only store recent events
            events = self._additional_events(session, tracker)

            for event in events:
                data = event.as_dict()

                intent = data.get("parse_data", {}).get("intent", {}).get("name")
                action = data.get("name")
                timestamp = data.get("timestamp")

                # noinspection PyArgumentList
                session.add(
                    self.SQLEvent(
                        sender_id=tracker.sender_id,
                        type_name=event.type_name,
                        timestamp=timestamp,
                        intent_name=intent,
                        action_name=action,
                        data=json.dumps(data),
                    )
                )
            session.commit()

        logger.debug(
            "Tracker with sender_id '{}' "
            "stored to database".format(tracker.sender_id)
        )

    def _additional_events(
        self, session: "Session", tracker: DialogueStateTracker
    ) -> Iterator:
        """Return events from the tracker which aren't currently stored."""

        n_events = (
            session.query(self.SQLEvent.sender_id)
            .filter_by(sender_id=tracker.sender_id)
            .count()
            or 0
        )

        return itertools.islice(tracker.events, n_events, len(tracker.events))
