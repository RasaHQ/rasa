import contextlib
import itertools
import json
import logging
import os
import pickle
import typing
from datetime import datetime, timezone

# noinspection PyPep8Naming
from time import sleep
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Text, Union

from boto3.dynamodb.conditions import Key
from rasa.core import utils
from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.brokers.broker import EventBroker
from rasa.core.conversation import Dialogue
from rasa.core.domain import Domain
from rasa.core.events import SessionStarted
from rasa.core.trackers import ActionExecuted, DialogueStateTracker, EventVerbosity
from rasa.utils.common import class_from_module_path, raise_warning, arguments_of
from rasa.utils.endpoints import EndpointConfig

if typing.TYPE_CHECKING:
    from sqlalchemy.engine.url import URL
    from sqlalchemy.engine.base import Engine
    from sqlalchemy.orm import Session, Query
    import boto3.resources.factory.dynamodb.Table

logger = logging.getLogger(__name__)


class TrackerStore:
    """Class to hold all of the TrackerStore classes"""

    def __init__(
        self, domain: Optional[Domain], event_broker: Optional[EventBroker] = None
    ) -> None:
        self.domain = domain
        self.event_broker = event_broker
        self.max_event_history = None

    @staticmethod
    def create(
        obj: Union["TrackerStore", EndpointConfig, None],
        domain: Optional[Domain] = None,
        event_broker: Optional[EventBroker] = None,
    ) -> "TrackerStore":
        """Factory to create a tracker store."""

        if isinstance(obj, TrackerStore):
            return obj
        else:
            return _create_from_endpoint_config(obj, domain, event_broker)

    @staticmethod
    def create_tracker_store(
        domain: Domain,
        store: Optional[EndpointConfig] = None,
        event_broker: Optional[EventBroker] = None,
    ) -> "TrackerStore":
        """Returns the tracker_store type"""

        raise_warning(
            "The `create_tracker_store` function is deprecated, please use "
            "`TrackerStore.create` instead. `create_tracker_store` will be "
            "removed in future Rasa versions.",
            DeprecationWarning,
        )

        return TrackerStore.create(store, domain, event_broker)

    def get_or_create_tracker(
        self,
        sender_id: Text,
        max_event_history: Optional[int] = None,
        append_action_listen: bool = True,
    ) -> "DialogueStateTracker":
        """Returns tracker or creates one if the retrieval returns None.

        Args:
            sender_id: Conversation ID associated with the requested tracker.
            max_event_history: Value to update the tracker store's max event history to.
            append_action_listen: Whether or not to append an initial `action_listen`.
        """
        tracker = self.retrieve(sender_id)
        self.max_event_history = max_event_history
        if tracker is None:
            tracker = self.create_tracker(
                sender_id, append_action_listen=append_action_listen
            )
        return tracker

    def init_tracker(self, sender_id: Text) -> "DialogueStateTracker":
        """Returns a Dialogue State Tracker"""
        return DialogueStateTracker(
            sender_id,
            self.domain.slots if self.domain else None,
            max_event_history=self.max_event_history,
        )

    def create_tracker(
        self, sender_id: Text, append_action_listen: bool = True
    ) -> DialogueStateTracker:
        """Creates a new tracker for `sender_id`.

        The tracker begins with a `SessionStarted` event and is initially listening.

        Args:
            sender_id: Conversation ID associated with the tracker.
            append_action_listen: Whether or not to append an initial `action_listen`.

        Returns:
            The newly created tracker for `sender_id`.

        """

        tracker = self.init_tracker(sender_id)

        if tracker:
            if append_action_listen:
                tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

            self.save(tracker)

        return tracker

    def save(self, tracker):
        """Save method that will be overridden by specific tracker"""
        raise NotImplementedError()

    def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Retrieve method that will be overridden by specific tracker"""
        raise NotImplementedError()

    def stream_events(self, tracker: DialogueStateTracker) -> None:
        """Streams events to a message broker"""
        offset = self.number_of_existing_events(tracker.sender_id)
        events = tracker.events
        for event in list(itertools.islice(events, offset, len(events))):
            body = {"sender_id": tracker.sender_id}
            body.update(event.as_dict())
            self.event_broker.publish(body)

    def number_of_existing_events(self, sender_id: Text) -> int:
        """Return number of stored events for a given sender id."""
        old_tracker = self.retrieve(sender_id)
        return len(old_tracker.events) if old_tracker else 0

    def keys(self) -> Iterable[Text]:
        """Returns the set of values for the tracker store's primary key"""
        raise NotImplementedError()

    @staticmethod
    def serialise_tracker(tracker: DialogueStateTracker) -> Text:
        """Serializes the tracker, returns representation of the tracker."""
        dialogue = tracker.as_dialogue()

        return json.dumps(dialogue.as_dict())

    @staticmethod
    def _deserialise_dialogue_from_pickle(
        sender_id: Text, serialised_tracker: bytes
    ) -> Dialogue:

        logger.warning(
            f"Found pickled tracker for "
            f"conversation ID '{sender_id}'. Deserialisation of pickled "
            f"trackers will be deprecated in version 2.0. Rasa will perform any "
            f"future save operations of this tracker using json serialisation."
        )
        return pickle.loads(serialised_tracker)

    def deserialise_tracker(
        self, sender_id: Text, serialised_tracker: Union[Text, bytes]
    ) -> Optional[DialogueStateTracker]:
        """Deserializes the tracker and returns it."""

        tracker = self.init_tracker(sender_id)
        if not tracker:
            return None

        try:
            dialogue = Dialogue.from_parameters(json.loads(serialised_tracker))
        except UnicodeDecodeError:
            dialogue = self._deserialise_dialogue_from_pickle(
                sender_id, serialised_tracker
            )

        tracker.recreate_from_dialogue(dialogue)

        return tracker


class InMemoryTrackerStore(TrackerStore):
    """Stores conversation history in memory"""

    def __init__(
        self, domain: Domain, event_broker: Optional[EventBroker] = None
    ) -> None:
        self.store = {}
        super().__init__(domain, event_broker)

    def save(self, tracker: DialogueStateTracker) -> None:
        """Updates and saves the current conversation state"""
        if self.event_broker:
            self.stream_events(tracker)
        serialised = InMemoryTrackerStore.serialise_tracker(tracker)
        self.store[tracker.sender_id] = serialised

    def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """
        Args:
            sender_id: the message owner ID

        Returns:
            DialogueStateTracker
        """
        if sender_id in self.store:
            logger.debug(f"Recreating tracker for id '{sender_id}'")
            return self.deserialise_tracker(sender_id, self.store[sender_id])
        else:
            logger.debug(f"Creating a new tracker for id '{sender_id}'.")
            return None

    def keys(self) -> Iterable[Text]:
        """Returns sender_ids of the Tracker Store in memory"""
        return self.store.keys()


class RedisTrackerStore(TrackerStore):
    """Stores conversation history in Redis"""

    def __init__(
        self,
        domain,
        host="localhost",
        port=6379,
        db=0,
        password: Optional[Text] = None,
        event_broker: Optional[EventBroker] = None,
        record_exp: Optional[float] = None,
        use_ssl: bool = False,
    ):
        import redis

        self.red = redis.StrictRedis(
            host=host, port=port, db=db, password=password, ssl=use_ssl
        )
        self.record_exp = record_exp
        super().__init__(domain, event_broker)

    def save(self, tracker, timeout=None):
        """Saves the current conversation state"""
        if self.event_broker:
            self.stream_events(tracker)

        if not timeout and self.record_exp:
            timeout = self.record_exp

        serialised_tracker = self.serialise_tracker(tracker)
        self.red.set(tracker.sender_id, serialised_tracker, ex=timeout)

    def retrieve(self, sender_id):
        """
        Args:
            sender_id: the message owner ID

        Returns:
            DialogueStateTracker
        """
        stored = self.red.get(sender_id)
        if stored is not None:
            return self.deserialise_tracker(sender_id, stored)
        else:
            return None

    def keys(self) -> Iterable[Text]:
        """Returns keys of the Redis Tracker Store"""
        return self.red.keys()


class DynamoTrackerStore(TrackerStore):
    """Stores conversation history in DynamoDB"""

    def __init__(
        self,
        domain: Domain,
        table_name: Text = "states",
        region: Text = "us-east-1",
        event_broker: Optional[EndpointConfig] = None,
    ):
        """
        Args:
            domain:
            table_name: The name of the DynamoDb table, does not
                need to be present a priori.
            event_broker:
        """
        import boto3

        self.client = boto3.client("dynamodb", region_name=region)
        self.region = region
        self.table_name = table_name
        self.db = self.get_or_create_table(table_name)
        super().__init__(domain, event_broker)

    def get_or_create_table(
        self, table_name: Text
    ) -> "boto3.resources.factory.dynamodb.Table":
        """Returns table or creates one if the table name is not in the table list"""
        import boto3

        dynamo = boto3.resource("dynamodb", region_name=self.region)
        if self.table_name not in self.client.list_tables()["TableNames"]:
            table = dynamo.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {"AttributeName": "sender_id", "KeyType": "HASH"},
                    {"AttributeName": "session_date", "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "sender_id", "AttributeType": "S"},
                    {"AttributeName": "session_date", "AttributeType": "N"},
                ],
                ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
            )

            # Wait until the table exists.
            table.meta.client.get_waiter("table_exists").wait(TableName=table_name)
        return dynamo.Table(table_name)

    def save(self, tracker):
        """Saves the current conversation state"""
        if self.event_broker:
            self.stream_events(tracker)
        self.db.put_item(Item=self.serialise_tracker(tracker))

    def serialise_tracker(self, tracker: "DialogueStateTracker") -> Dict:
        """Serializes the tracker, returns object with decimal types"""
        d = tracker.as_dialogue().as_dict()
        d.update(
            {
                "sender_id": tracker.sender_id,
                "session_date": int(datetime.now(tz=timezone.utc).timestamp()),
            }
        )
        return utils.replace_floats_with_decimals(d)

    def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Create a tracker from all previously stored events."""

        # Retrieve dialogues for a sender_id in reverse chronological order based on
        # the session_date sort key
        dialogues = self.db.query(
            KeyConditionExpression=Key("sender_id").eq(sender_id),
            Limit=1,
            ScanIndexForward=False,
        )["Items"]
        if dialogues:
            return DialogueStateTracker.from_dict(
                sender_id, dialogues[0].get("events"), self.domain.slots
            )
        else:
            return None

    def keys(self) -> Iterable[Text]:
        """Returns sender_ids of the DynamoTrackerStore"""
        return [
            i["sender_id"]
            for i in self.db.scan(ProjectionExpression="sender_id")["Items"]
        ]


class MongoTrackerStore(TrackerStore):
    """
    Stores conversation history in Mongo

    Property methods:
        conversations: returns the current conversation
    """

    def __init__(
        self,
        domain: Domain,
        host: Optional[Text] = "mongodb://localhost:27017",
        db: Optional[Text] = "rasa",
        username: Optional[Text] = None,
        password: Optional[Text] = None,
        auth_source: Optional[Text] = "admin",
        collection: Optional[Text] = "conversations",
        event_broker: Optional[EventBroker] = None,
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
        super().__init__(domain, event_broker)

        self._ensure_indices()

    @property
    def conversations(self):
        """Returns the current conversation"""
        return self.db[self.collection]

    def _ensure_indices(self):
        """Create an index on the sender_id"""
        self.conversations.create_index("sender_id")

    @staticmethod
    def _current_tracker_state_without_events(tracker: DialogueStateTracker) -> Dict:
        # get current tracker state and remove `events` key from state
        # since events are pushed separately in the `update_one()` operation
        state = tracker.current_state(EventVerbosity.ALL)
        state.pop("events", None)

        return state

    def save(self, tracker, timeout=None):
        """Saves the current conversation state"""
        if self.event_broker:
            self.stream_events(tracker)

        additional_events = self._additional_events(tracker)

        self.conversations.update_one(
            {"sender_id": tracker.sender_id},
            {
                "$set": self._current_tracker_state_without_events(tracker),
                "$push": {
                    "events": {"$each": [e.as_dict() for e in additional_events]}
                },
            },
            upsert=True,
        )

    def _additional_events(self, tracker: DialogueStateTracker) -> Iterator:
        """Return events from the tracker which aren't currently stored.

        Args:
            tracker: Tracker to inspect.

        Returns:
            List of serialised events that aren't currently stored.

        """

        stored = self.conversations.find_one({"sender_id": tracker.sender_id}) or {}
        number_events_since_last_session = len(
            self._events_since_last_session_start(stored)
        )

        return itertools.islice(
            tracker.events, number_events_since_last_session, len(tracker.events)
        )

    @staticmethod
    def _events_since_last_session_start(serialised_tracker: Dict) -> List[Dict]:
        """Retrieve events since and including the latest `SessionStart` event.

        Args:
            serialised_tracker: Serialised tracker to inspect.

        Returns:
            List of serialised events since and including the latest `SessionStarted`
            event. Returns all events if no such event is found.

        """

        events = []
        for event in reversed(serialised_tracker.get("events", [])):
            events.append(event)
            if event["event"] == SessionStarted.type_name:
                break

        return list(reversed(events))

    def retrieve(self, sender_id):
        """
        Args:
            sender_id: the message owner ID

        Returns:
            `DialogueStateTracker`
        """
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
            events = self._events_since_last_session_start(stored)
            return DialogueStateTracker.from_dict(sender_id, events, self.domain.slots)
        else:
            return None

    def keys(self) -> Iterable[Text]:
        """Returns sender_ids of the Mongo Tracker Store"""
        return [c["sender_id"] for c in self.conversations.find()]


class SQLTrackerStore(TrackerStore):
    """Store which can save and retrieve trackers from an SQL database."""

    from sqlalchemy.ext.declarative import declarative_base

    Base = declarative_base()

    class SQLEvent(Base):
        """Represents an event in the SQL Tracker Store"""

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
        event_broker: Optional[EventBroker] = None,
        login_db: Optional[Text] = None,
        query: Optional[Dict] = None,
    ) -> None:
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import create_engine
        import sqlalchemy.exc

        engine_url = self.get_db_url(
            dialect, host, port, db, username, password, login_db, query
        )
        logger.debug(f"Attempting to connect to database via '{engine_url}'.")

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
                    logger.error(f"Could not create tables: {e}")

                self.sessionmaker = sessionmaker(bind=self.engine)
                break
            except (
                sqlalchemy.exc.OperationalError,
                sqlalchemy.exc.IntegrityError,
            ) as error:

                logger.warning(error)
                sleep(5)

        logger.debug(f"Connection to SQL database '{db}' successful.")

        super().__init__(domain, event_broker)

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
        """Create databse `db` and update engine to reflect the updated `engine_url`."""

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
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db}'")
        exists = cursor.fetchone()
        if not exists:
            try:
                cursor.execute(f"CREATE DATABASE {db}")
            except psycopg2.IntegrityError as e:
                logger.error(f"Could not create database '{db}': {e}")

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
        """Returns sender_ids of the SQLTrackerStore"""
        with self.session_scope() as session:
            sender_ids = session.query(self.SQLEvent.sender_id).distinct().all()
            return [sender_id for (sender_id,) in sender_ids]

    def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Create a tracker from all previously stored events."""

        with self.session_scope() as session:

            serialised_events = self._event_query(session, sender_id).all()

            events = [json.loads(event.data) for event in serialised_events]

            if self.domain and len(events) > 0:
                logger.debug(f"Recreating tracker from sender id '{sender_id}'")
                return DialogueStateTracker.from_dict(
                    sender_id, events, self.domain.slots
                )
            else:
                logger.debug(
                    f"Can't retrieve tracker matching "
                    f"sender id '{sender_id}' from SQL storage. "
                    f"Returning `None` instead."
                )
                return None

    def _event_query(self, session: "Session", sender_id: Text) -> "Query":
        """Provide the query to retrieve the conversation events for a specific sender.

        Args:
            session: Current database session.
            sender_id: Sender id whose conversation events should be retrieved.

        Returns:
            Query to get the conversation events.
        """
        import sqlalchemy as sa

        # Subquery to find the timestamp of the latest `SessionStarted` event
        session_start_sub_query = (
            session.query(sa.func.max(self.SQLEvent.timestamp).label("session_start"))
            .filter(
                self.SQLEvent.sender_id == sender_id,
                self.SQLEvent.type_name == SessionStarted.type_name,
            )
            .subquery()
        )

        return (
            session.query(self.SQLEvent)
            .filter(
                self.SQLEvent.sender_id == sender_id,
                # Find events after the latest `SessionStarted` event or return all
                # events
                sa.or_(
                    self.SQLEvent.timestamp >= session_start_sub_query.c.session_start,
                    session_start_sub_query.c.session_start.is_(None),
                ),
            )
            .order_by(self.SQLEvent.timestamp)
        )

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

        logger.debug(f"Tracker with sender_id '{tracker.sender_id}' stored to database")

    def _additional_events(
        self, session: "Session", tracker: DialogueStateTracker
    ) -> Iterator:
        """Return events from the tracker which aren't currently stored."""

        number_of_events_since_last_session = self._event_query(
            session, tracker.sender_id
        ).count()
        return itertools.islice(
            tracker.events, number_of_events_since_last_session, len(tracker.events)
        )


class FailSafeTrackerStore(TrackerStore):
    """Wraps a tracker store so that we can fallback to a different tracker store in
    case of errors."""

    def __init__(
        self,
        tracker_store: TrackerStore,
        on_tracker_store_error: Optional[Callable[[Exception], None]] = None,
        fallback_tracker_store: Optional[TrackerStore] = None,
    ) -> None:
        """Create a `FailSafeTrackerStore`.

        Args:
            tracker_store: Primary tracker store.
            on_tracker_store_error: Callback which is called when there is an error
                in the primary tracker store.
        """

        self._fallback_tracker_store: Optional[TrackerStore] = fallback_tracker_store
        self._tracker_store = tracker_store
        self._on_tracker_store_error = on_tracker_store_error

        super().__init__(tracker_store.domain, tracker_store.event_broker)

    @property
    def domain(self) -> Optional[Domain]:
        return self._tracker_store.domain

    @domain.setter
    def domain(self, domain: Optional[Domain]) -> None:
        self._tracker_store.domain = domain

        if self._fallback_tracker_store:
            self._fallback_tracker_store.domain = domain

    @property
    def fallback_tracker_store(self) -> TrackerStore:
        if not self._fallback_tracker_store:
            self._fallback_tracker_store = InMemoryTrackerStore(
                self._tracker_store.domain, self._tracker_store.event_broker
            )

        return self._fallback_tracker_store

    def on_tracker_store_error(self, error: Exception) -> None:
        if self._on_tracker_store_error:
            self._on_tracker_store_error(error)
        else:
            logger.error(
                f"Error happened when trying to save conversation tracker to "
                f"'{self._tracker_store.__class__.__name__}'. Falling back to use "
                f"the '{InMemoryTrackerStore.__name__}'. Please "
                f"investigate the following error: {error}."
            )

    def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        try:
            return self._tracker_store.retrieve(sender_id)
        except Exception as e:
            self.on_tracker_store_error(e)
            return None

    def keys(self) -> Iterable[Text]:
        try:
            return self._tracker_store.keys()
        except Exception as e:
            self.on_tracker_store_error(e)
            return []

    def save(self, tracker: DialogueStateTracker) -> None:
        try:
            self._tracker_store.save(tracker)
        except Exception as e:
            self.on_tracker_store_error(e)
            self.fallback_tracker_store.save(tracker)


def _create_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig] = None,
    domain: Optional[Domain] = None,
    event_broker: Optional[EventBroker] = None,
) -> "TrackerStore":
    """Given an endpoint configuration, create a proper tracker store object."""

    domain = domain or Domain.empty()

    if endpoint_config is None or endpoint_config.type is None:
        # default tracker store if no type is set
        tracker_store = InMemoryTrackerStore(domain, event_broker)
    elif endpoint_config.type.lower() == "redis":
        tracker_store = RedisTrackerStore(
            domain=domain,
            host=endpoint_config.url,
            event_broker=event_broker,
            **endpoint_config.kwargs,
        )
    elif endpoint_config.type.lower() == "mongod":
        tracker_store = MongoTrackerStore(
            domain=domain,
            host=endpoint_config.url,
            event_broker=event_broker,
            **endpoint_config.kwargs,
        )
    elif endpoint_config.type.lower() == "sql":
        tracker_store = SQLTrackerStore(
            domain=domain,
            host=endpoint_config.url,
            event_broker=event_broker,
            **endpoint_config.kwargs,
        )
    elif endpoint_config.type.lower() == "dynamo":
        tracker_store = DynamoTrackerStore(
            domain=domain, event_broker=event_broker, **endpoint_config.kwargs
        )
    else:
        tracker_store = _load_from_module_string(domain, endpoint_config, event_broker)

    logger.debug(f"Connected to {tracker_store.__class__.__name__}.")

    return tracker_store


def _load_from_module_string(
    domain: Domain, store: EndpointConfig, event_broker: Optional[EventBroker] = None
) -> "TrackerStore":
    """Initializes a custom tracker.

    Defaults to the InMemoryTrackerStore if the module path can not be found.

    Args:
        domain: defines the universe in which the assistant operates
        store: the specific tracker store
        event_broker: an event broker to publish events

    Returns:
        a tracker store from a specified type in a stores endpoint configuration
    """

    try:
        tracker_store_class = class_from_module_path(store.type)
        init_args = arguments_of(tracker_store_class.__init__)
        if "url" in init_args and "host" not in init_args:
            raise_warning(
                "The `url` initialization argument for custom tracker stores is "
                "deprecated. Your custom tracker store should take a `host` "
                "argument in its `__init__()` instead.",
                DeprecationWarning,
            )
            store.kwargs["url"] = store.url
        else:
            store.kwargs["host"] = store.url

        return tracker_store_class(
            domain=domain, event_broker=event_broker, **store.kwargs
        )
    except (AttributeError, ImportError):
        raise_warning(
            f"Tracker store with type '{store.type}' not found. "
            f"Using `InMemoryTrackerStore` instead."
        )
        return InMemoryTrackerStore(domain)
