from __future__ import annotations

import contextlib
import itertools
import json
import os
from datetime import datetime
from time import sleep
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    Iterator,
    Optional,
    Text,
    Union,
)

import sqlalchemy as sa
import structlog

import rasa.shared
from rasa.core.brokers.broker import EventBroker
from rasa.core.constants import (
    POSTGRESQL_MAX_OVERFLOW,
    POSTGRESQL_POOL_SIZE,
    POSTGRESQL_SCHEMA,
    SQL_SERVICE_NAME,
    SQL_TRACKER_STORE_SSL_MODE_ENV_VAR_NAME,
    SQL_TRACKER_STORE_SSL_ROOT_CERTIFICATE_ENV_VAR_NAME,
)
from rasa.core.iam_credentials_providers.credentials_provider_protocol import (
    IAMCredentialsProviderInput,
    SupportedServiceType,
    create_iam_credentials_provider,
)
from rasa.core.tracker_stores.tracker_store import (
    SerializedTrackerAsText,
    TrackerStore,
    validate_port,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SessionStarted
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.constants import INTENT_NAME_KEY

if TYPE_CHECKING:
    from sqlalchemy import Sequence
    from sqlalchemy.engine.base import Engine
    from sqlalchemy.engine.url import URL
    from sqlalchemy.orm import Query, Session


structlogger = structlog.get_logger(__name__)

# default values of PostgreSQL pool size and max overflow
POSTGRESQL_DEFAULT_MAX_OVERFLOW = 100
POSTGRESQL_DEFAULT_POOL_SIZE = 50


def _create_sequence(table_name: Text) -> "Sequence":
    """Creates a sequence object for a specific table name.

    If using Oracle you will need to create a sequence in your database,
    as described here: https://rasa.com/docs/rasa-pro/production/tracker-stores#sqltrackerstore
    Args:
        table_name: The name of the table, which gets a Sequence assigned

    Returns: A `Sequence` object
    """
    from sqlalchemy.orm import declarative_base

    sequence_name = f"{table_name}_seq"
    Base = declarative_base()
    return sa.Sequence(sequence_name, metadata=Base.metadata, optional=True)


def is_postgresql_url(url: Union[Text, "URL"]) -> bool:
    """Determine whether `url` configures a PostgreSQL connection.

    Args:
        url: SQL connection URL.

    Returns:
        `True` if `url` is a PostgreSQL connection URL.
    """
    if isinstance(url, str):
        return "postgresql" in url

    return url.drivername == "postgresql"


def get_ssl_args() -> Dict[str, Any]:
    """Get SSL arguments for PostgreSQL connection from environment variables."""
    ssl_mode = os.getenv(SQL_TRACKER_STORE_SSL_MODE_ENV_VAR_NAME)
    ssl_root_cert = os.getenv(SQL_TRACKER_STORE_SSL_ROOT_CERTIFICATE_ENV_VAR_NAME)
    ssl_args = {}

    if ssl_mode:
        ssl_args["sslmode"] = ssl_mode

    if ssl_root_cert:
        ssl_args["sslrootcert"] = ssl_root_cert

    return ssl_args


def create_engine_kwargs(url: Union[Text, "URL"]) -> Dict[Text, Any]:
    """Get `sqlalchemy.create_engine()` kwargs.

    Args:
        url: SQL connection URL.

    Returns:
        kwargs to be passed into `sqlalchemy.create_engine()`.
    """
    if not is_postgresql_url(url):
        return {}

    kwargs: Dict[Text, Any] = {}

    schema_name = os.environ.get(POSTGRESQL_SCHEMA)

    if schema_name:
        structlogger.debug(
            "postgresql_tracker_store.schema_name",
            event_inf=f"Using PostgreSQL schema '{schema_name}'.",
        )
        kwargs["connect_args"] = {"options": f"-csearch_path={schema_name}"}

    # pool_size and max_overflow can be set to control the number of
    # connections that are kept in the connection pool. Not available
    # for SQLite, and only  tested for PostgreSQL. See
    # https://docs.sqlalchemy.org/en/13/core/pooling.html#sqlalchemy.pool.QueuePool
    kwargs["pool_size"] = int(
        os.environ.get(POSTGRESQL_POOL_SIZE, POSTGRESQL_DEFAULT_POOL_SIZE)
    )
    kwargs["max_overflow"] = int(
        os.environ.get(POSTGRESQL_MAX_OVERFLOW, POSTGRESQL_DEFAULT_MAX_OVERFLOW)
    )

    ssl_args = get_ssl_args()

    if ssl_args:
        if "connect_args" in kwargs:
            kwargs["connect_args"].update(ssl_args)
        else:
            kwargs["connect_args"] = ssl_args

    return kwargs


def ensure_schema_exists(session: "Session") -> None:
    """Ensure that the requested PostgreSQL schema exists in the database.

    Args:
        session: Session used to inspect the database.

    Raises:
        `ValueError` if the requested schema does not exist.
        RasaException if no engine can be obtained from session.
    """
    schema_name = os.environ.get(POSTGRESQL_SCHEMA)

    if not schema_name:
        return

    engine = session.get_bind()

    if not isinstance(engine, sa.engine.base.Engine):
        # The "bind" is usually an instance of Engine, except in the case
        # where the session has been explicitly bound directly to a connection.
        raise RasaException("Cannot ensure schema exists as no engine exists.")

    if is_postgresql_url(engine.url):
        query = sa.exists(
            sa.select(sa.text("schema_name"))
            .select_from(sa.text("information_schema.schemata"))
            .where(sa.text(f"schema_name = '{schema_name}'"))
        )
        if not session.query(query).scalar():
            raise ValueError(schema_name)


class SQLTrackerStore(TrackerStore, SerializedTrackerAsText):
    """Store which can save and retrieve trackers from an SQL database."""

    from sqlalchemy.orm import DeclarativeBase

    class Base(DeclarativeBase):
        """Base class for all tracker store tables."""

        pass

    class SQLEvent(Base):
        """Represents an event in the SQL Tracker Store."""

        __tablename__ = "events"

        # `create_sequence` is needed to create a sequence for databases that
        # don't autoincrement Integer primary keys (e.g. Oracle)
        id = sa.Column(sa.Integer, _create_sequence(__tablename__), primary_key=True)
        sender_id = sa.Column(sa.String(255), nullable=False, index=True)
        type_name = sa.Column(sa.String(255), nullable=False)
        timestamp = sa.Column(sa.Float)
        intent_name = sa.Column(sa.String(255))
        action_name = sa.Column(sa.String(255))
        data = sa.Column(sa.Text)

    def __init__(
        self,
        domain: Optional[Domain] = None,
        dialect: Text = "sqlite",
        host: Optional[Text] = None,
        port: Optional[int] = None,
        db: Text = "rasa.db",
        username: Optional[Text] = None,
        password: Optional[Text] = None,
        event_broker: Optional[EventBroker] = None,
        login_db: Optional[Text] = None,
        query: Optional[Dict] = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        import sqlalchemy.exc

        port = validate_port(port)

        iam_credentials_provider = create_iam_credentials_provider(
            IAMCredentialsProviderInput(
                service_type=SupportedServiceType.TRACKER_STORE,
                service_name=SQL_SERVICE_NAME,
                username=username,
                host=host,
                port=port,
            )
        )
        if iam_credentials_provider is not None:
            credentials = iam_credentials_provider.get_temporary_credentials()
            if credentials.auth_token:
                password = credentials.auth_token
                structlogger.debug(
                    "sql_tracker_store.iam_credentials_provider",
                    event_info="Using temporary auth token from "
                    "IAM credentials provider.",
                )
            else:
                structlogger.warning(
                    "sql_tracker_store.iam_credentials_provider.no_auth_token",
                    event_info=(
                        "IAM credentials provider did not return an auth token. "
                        "Falling back to provided password or no password."
                    ),
                )

        engine_url = self.get_db_url(
            dialect, host, port, db, username, password, login_db, query
        )

        self.engine = sa.create_engine(engine_url, **create_engine_kwargs(engine_url))

        structlogger.debug(
            "sql_tracker_store.connect_to_sql_database",
            event_info=f"Attempting to connect to database via '{self.engine.url!r}'.",
        )

        # Database might take a while to come up
        while True:
            try:
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
                    structlogger.error(
                        "sql_tracker_store.create_tables_failed",
                        event_info="Could not create tables",
                        exec_info=e,
                    )

                self.sessionmaker = sa.orm.session.sessionmaker(bind=self.engine)
                break
            except (
                sqlalchemy.exc.OperationalError,
                sqlalchemy.exc.IntegrityError,
            ) as error:
                structlogger.warning(
                    "sql_tracker_store.initialisation_error",
                    event_info="Failed to establish a connection to the SQL database. ",
                    exc_info=error,
                )
                sleep(5)

        structlogger.debug(
            "sql_tracker_store.connected_to_sql_database",
            event_info=f"Connection to SQL database '{db}' successful.",
        )

        super().__init__(domain, event_broker, **kwargs)

    @staticmethod
    def get_db_url(
        dialect: Text = "sqlite",
        host: Optional[Text] = None,
        port: Optional[int] = None,
        db: Text = "rasa.db",
        username: Optional[Text] = None,
        password: Optional[Text] = None,
        login_db: Optional[Text] = None,
        query: Optional[Dict] = None,
    ) -> Union[Text, "URL"]:
        """Build an SQLAlchemy `URL` object.

        The URL object represents the parameters needed to connect to an
        SQL database.

        Args:
            dialect: SQL database type.
            host: Database network host.
            port: Database network port.
            db: Database name.
            username: Username to use when connecting to the database.
            password: Password for database user.
            login_db: Alternative database name to which initially connect, and create
                the database specified by `db` (PostgreSQL only).
            query: Dictionary of options to be passed to the dialect and/or the
                DBAPI upon connect.

        Returns:
            URL ready to be used with an SQLAlchemy `Engine` object.
        """
        from urllib import parse

        # Users might specify a url in the host
        if host and "://" in host:
            # assumes this is a complete database host name including
            # e.g. `postgres://...`
            return host
        elif host:
            # add fake scheme to properly parse components
            parsed = parse.urlsplit(f"scheme://{host}")

            # users might include the port in the url
            port = parsed.port or port
            host = parsed.hostname or host

        if not query:
            # query needs to be set in order to create a URL
            query = {}

        return sa.engine.url.URL(
            dialect,
            username,
            password,
            host,
            port,
            database=login_db if login_db else db,
            query=query,
        )

    def _create_database_and_update_engine(self, db: Text, engine_url: "URL") -> None:
        """Creates database `db` and updates engine accordingly."""
        from sqlalchemy import create_engine

        if self.engine.dialect.name != "postgresql":
            rasa.shared.utils.io.raise_warning(
                "The parameter 'login_db' can only be used with a postgres database."
            )
            return

        self._create_database(self.engine, db)
        self.engine.dispose()
        engine_url = sa.engine.url.URL(
            drivername=engine_url.drivername,
            username=engine_url.username,
            password=engine_url.password,
            host=engine_url.host,
            port=engine_url.port,
            database=db,
            query=engine_url.query,
        )
        self.engine = create_engine(engine_url)

    @staticmethod
    def _create_database(engine: "Engine", database_name: Text) -> None:
        """Create database `db` on `engine` if it does not exist."""
        import sqlalchemy.exc

        with engine.connect() as connection:
            connection.execution_options(isolation_level="AUTOCOMMIT")
            matching_rows = connection.execute(
                sa.text(
                    f"SELECT 1 FROM pg_catalog.pg_database "
                    f"WHERE datname = '{database_name}'"
                )
            ).rowcount

            if not matching_rows:
                try:
                    connection.execute(sa.text(f"CREATE DATABASE {database_name}"))
                except (
                    sqlalchemy.exc.ProgrammingError,
                    sqlalchemy.exc.IntegrityError,
                ) as e:
                    structlogger.error(
                        "sql_tracker_store.create_database_failed",
                        event_info=f"Could not create database '{database_name}'",
                        exec_info=e,
                    )

    @contextlib.contextmanager
    def session_scope(self) -> Generator["Session", None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self.sessionmaker()
        try:
            ensure_schema_exists(session)
            yield session
        except ValueError as e:
            rasa.shared.utils.cli.print_error_and_exit(
                f"Requested PostgreSQL schema '{e}' was not found in the database. To "
                f"continue, please create the schema by running 'CREATE DATABASE {e};' "
                f"or unset the '{POSTGRESQL_SCHEMA}' environment variable in order to "
                f"use the default schema. Exiting application."
            )
        finally:
            session.close()

    async def keys(self) -> Iterable[Text]:
        """Returns sender_ids of the SQLTrackerStore."""
        with self.session_scope() as session:
            sender_ids = session.query(self.SQLEvent.sender_id).distinct().all()
            return [sender_id for (sender_id,) in sender_ids]

    async def delete(self, sender_id: Text) -> None:
        """Delete tracker for the given sender_id."""
        if not await self.exists(sender_id):
            structlogger.info(
                "sql_tracker_store.delete.no_tracker_for_sender_id",
                event_info=f"Could not find tracker for conversation ID '{sender_id}'.",
            )
            return None

        with self.session_scope() as session:
            statement = sa.delete(self.SQLEvent).where(
                self.SQLEvent.sender_id == sender_id
            )
            result = session.execute(statement)
            session.commit()

        structlogger.info(
            "sql_tracker_store.delete.deleted_tracker",
            sender_id=sender_id,
            num_rows=result.rowcount,
        )

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Retrieves tracker for the latest conversation session."""
        return await self._retrieve(sender_id, fetch_events_from_all_sessions=False)

    async def retrieve_full_tracker(
        self, conversation_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Fetching all tracker events across conversation sessions."""
        return await self._retrieve(
            conversation_id, fetch_events_from_all_sessions=True
        )

    async def count_conversations(self, after_timestamp: float = 0.0) -> int:
        """Returns the number of conversations that have occurred after a timestamp.

        By default, this method returns the number of conversations that
        have occurred after the Unix epoch (i.e. timestamp 0).
        """
        with self.session_scope() as session:
            query = (
                session.query(self.SQLEvent.sender_id)
                .distinct()
                .filter(self.SQLEvent.timestamp >= after_timestamp)
            )
            return query.count()

    async def _retrieve(
        self, sender_id: Text, fetch_events_from_all_sessions: bool
    ) -> Optional[DialogueStateTracker]:
        with self.session_scope() as session:
            serialised_events = self._event_query(
                session,
                sender_id,
                fetch_events_from_all_sessions=fetch_events_from_all_sessions,
            ).all()

            events = [json.loads(event.data) for event in serialised_events]

            if self.domain and len(events) > 0:
                structlogger.debug(
                    "sql_tracker_store.recreating_tracker",
                    event_info=f"Recreating tracker from sender id '{sender_id}'",
                )
                return DialogueStateTracker.from_dict(
                    sender_id, events, self.domain.slots
                )
            else:
                structlogger.debug(
                    "sql_tracker_store._retrieve.no_tracker_for_sender_id",
                    event_info=(
                        f"Can't retrieve tracker matching "
                        f"sender id '{sender_id}' from SQL storage. "
                        f"Returning `None` instead.",
                    ),
                )
                return None

    def _event_query(
        self, session: "Session", sender_id: Text, fetch_events_from_all_sessions: bool
    ) -> "Query":
        """Provide the query to retrieve the conversation events for a specific sender.

        The events are ordered by ID to ensure correct sequence of events.
        As `timestamp` is not guaranteed to be unique and low-precision (float), it
        cannot be used to order the events.

        Args:
            session: Current database session.
            sender_id: Sender id whose conversation events should be retrieved.
            fetch_events_from_all_sessions: Whether to fetch events from all
                conversation sessions. If `False`, only fetch events from the
                latest conversation session.

        Returns:
            Query to get the conversation events.
        """
        # Subquery to find the timestamp of the latest `SessionStarted` event
        session_start_sub_query = (
            session.query(sa.func.max(self.SQLEvent.timestamp).label("session_start"))
            .filter(
                self.SQLEvent.sender_id == sender_id,
                self.SQLEvent.type_name == SessionStarted.type_name,
            )
            .subquery()
        )

        event_query = session.query(self.SQLEvent).filter(
            self.SQLEvent.sender_id == sender_id
        )
        if not fetch_events_from_all_sessions:
            event_query = event_query.filter(
                # Find events after the latest `SessionStarted` event or return all
                # events
                sa.or_(
                    self.SQLEvent.timestamp >= session_start_sub_query.c.session_start,
                    session_start_sub_query.c.session_start.is_(None),
                )
            )

        return event_query.order_by(self.SQLEvent.id)

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Update database with events from the current conversation."""
        await self.stream_events(tracker)

        with self.session_scope() as session:
            # only store recent events
            events = self._additional_events(session, tracker)

            for event in events:
                data = event.as_dict()
                intent = (
                    data.get("parse_data", {}).get("intent", {}).get(INTENT_NAME_KEY)
                )
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

        structlogger.debug(
            "sql_tracker_store.save_tracker",
            event_info=(
                f"Tracker with sender_id '{tracker.sender_id}' stored to database",
            ),
        )

    def _additional_events(
        self, session: "Session", tracker: DialogueStateTracker
    ) -> Iterator:
        """Return events from the tracker which aren't currently stored."""
        number_of_events_since_last_session = self._event_query(
            session, tracker.sender_id, fetch_events_from_all_sessions=False
        ).count()

        return itertools.islice(
            tracker.events, number_of_events_since_last_session, len(tracker.events)
        )

    async def update(self, tracker_to_keep: DialogueStateTracker) -> None:
        """Overwrite the tracker in the SQL tracker store."""
        with self.session_scope() as session:
            # Delete events whose timestamp are older
            # than the first event of the tracker to keep.
            statement = sa.delete(self.SQLEvent).where(
                self.SQLEvent.sender_id == tracker_to_keep.sender_id,
                self.SQLEvent.timestamp < tracker_to_keep.events[0].timestamp
                if tracker_to_keep.events
                else 0,
            )

            result = session.execute(statement)
            session.commit()

        first_event_timestamp = str(
            datetime.fromtimestamp(tracker_to_keep.events[0].timestamp)
        )

        structlogger.info(
            "sql_tracker_store.update.updated_tracker",
            sender_id=tracker_to_keep.sender_id,
            first_event_timestamp=first_event_timestamp,
            event_info=f"{result.rowcount} rows removed from tracker.",
        )
