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
    List,
    Optional,
    Text,
    Union,
    cast,
)

import sqlalchemy as sa
import structlog
from jsonpatch import JsonPatchException
from jsonpointer import JsonPointerException

import rasa.shared
from rasa.constants import DEFAULT_SANIC_WORKERS, ENV_SANIC_WORKERS, USER_ID
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
from rasa.shared.core.constants import ACTION_SESSION_START_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, Event
from rasa.shared.core.trackers import (
    DialogueStateTracker,
    get_latest_replay_safe_session_tracker,
)
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
    """Store which can save and retrieve trackers from an SQL database.

    Latest-session retrieval and incremental saves align on the same session
    boundary as other tracker stores: ``ActionExecuted(action_session_start)``.
    That action is always run by :class:`~rasa.core.processor.MessageProcessor`
    when a new session starts (empty conversation or session expiry with
    ``start_session_after_expiry``), so it is a stable marker in persisted
    histories. ``SessionStarted`` is still emitted by the default
    :class:`~rasa.core.actions.action.ActionSessionStart` implementation but can
    be omitted by custom session-start actions; using ``action_session_start``
    avoids depending on that optional event for slicing.
    """

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

    class SQLUser(Base):
        """Mapping table between user_id and sender_id for efficient querying.

        This table enables efficient queries to find all trackers (sender_ids)
        associated with a given user_id using JOIN operations instead of
        scanning all events.

        Note: user_id can be NULL for anonymous users. Rows with NULL user_id
        are not stored in this table (only authenticated users are tracked).
        """

        __tablename__ = "users"

        sender_id = sa.Column(
            sa.String(255),
            primary_key=True,
            nullable=False,
            index=True,
        )
        user_id = sa.Column(
            sa.String(255),
            nullable=False,
            index=True,  # Index for efficient user_id lookups
        )
        conversation_started_timestamp = sa.Column(
            sa.Float,
            nullable=True,
            index=True,  # Index for efficient sorting by timestamp
        )

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
                    self._create_tables(dialect)
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

        # Cached result of Inspector.has_table("users"). None means "not yet resolved".
        # The schema is stable after startup, so this is safe to cache for the
        # lifetime of the instance.
        self._has_users_table: Optional[bool] = None
        super().__init__(domain, event_broker, **kwargs)

    @property
    def _users_table_exists(self) -> bool:
        """Return whether the ``users`` table exists in the database.

        The result is cached on first access because the schema is stable after startup.
        Two concurrent workers racing on initialisation both derive the same idempotent
        answer; no lock is needed.
        """
        if self._has_users_table is None:
            self._has_users_table = sa.inspect(self.engine).has_table("users")
        return self._has_users_table

    def _create_tables(
        self,
        dialect: Text = "sqlite",
    ) -> None:
        """Create database tables with appropriate locking mechanism.

        Uses advisory locks for PostgreSQL when multiple Sanic workers are configured
        to prevent race conditions during concurrent initialization.
        """
        sanic_workers = int(os.environ.get(ENV_SANIC_WORKERS, DEFAULT_SANIC_WORKERS))

        if sanic_workers > 1:
            if dialect == "postgresql":
                self._create_tables_with_advisory_lock()
            else:
                structlogger.warning(
                    "sql_tracker_store.multiple_workers_without_locking",
                    event_info=(
                        "Advisory lock mechanism is not supported for non-PostgreSQL "
                        "databases when using multiple Sanic workers. Running with "
                        f"{sanic_workers} Sanic workers using {dialect} database may"
                        "result in race conditions during concurrent table creation."
                    ),
                )
                self.Base.metadata.create_all(self.engine)
        else:
            self.Base.metadata.create_all(self.engine)

    def _create_tables_with_advisory_lock(self) -> None:
        """Create tables using PostgreSQL advisory lock to prevent race conditions.

        Multiple Sanic workers may attempt to create tables simultaneously. The advisory
        lock ensures only one worker creates the schema while others wait.
        """
        from sqlalchemy.engine import Inspector

        # Use a hash of the table name as lock ID to ensure all workers
        # use the same lock.
        # Modulo keeps it within PostgreSQL's int range.
        lock_id = hash("events") % (2**31)

        with self.engine.begin() as conn:
            # Acquire advisory lock - blocks until available
            conn.execute(sa.text(f"SELECT pg_advisory_lock({lock_id})"))
            try:
                # Double-check if tables exist before creating
                inspector = Inspector.from_engine(self.engine)
                if not inspector.has_table("events") or not inspector.has_table(
                    "users"
                ):
                    self.Base.metadata.create_all(self.engine, checkfirst=True)
                    structlogger.debug(
                        "sql_tracker_store.tables_created",
                        event_info="Successfully created database tables.",
                    )
                else:
                    structlogger.debug(
                        "sql_tracker_store.tables_already_exist",
                        event_info="Tables already exist, skipping creation.",
                    )
            finally:
                # Always release the lock
                conn.execute(sa.text(f"SELECT pg_advisory_unlock({lock_id})"))

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
            # Delete events
            statement = sa.delete(self.SQLEvent).where(
                self.SQLEvent.sender_id == sender_id
            )
            result = session.execute(statement)
            if not isinstance(result, sa.engine.cursor.CursorResult):
                result = cast(sa.engine.cursor.CursorResult, result)

            # Clean up users table
            user_statement = sa.delete(self.SQLUser).where(
                self.SQLUser.sender_id == sender_id
            )
            session.execute(user_statement)

            session.commit()

        structlogger.info(
            "sql_tracker_store.delete.deleted_tracker",
            sender_id=sender_id,
            num_rows=result.rowcount,
        )

    async def retrieve(self, sender_id: str) -> Optional[DialogueStateTracker]:
        """Retrieves tracker for the latest conversation session.

        The latest session is the replay-safe slice from the last
        ``action_session_start`` action (see
        :func:`get_latest_replay_safe_session_tracker`).
        """
        tracker = await self._retrieve(sender_id, fetch_events_from_all_sessions=True)
        if tracker is None:
            return None

        return get_latest_replay_safe_session_tracker(
            tracker,
            start_session_after_expiry=(
                self.domain.session_config.start_session_after_expiry
            ),
        )

    async def retrieve_full_tracker(
        self, conversation_id: str
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
                tracker = DialogueStateTracker.from_dict(
                    sender_id, events, self.domain.slots
                )

                if self._users_table_exists:
                    user_mapping = (
                        session.query(self.SQLUser.user_id)
                        .filter(self.SQLUser.sender_id == sender_id)
                        .one_or_none()
                    )
                    if user_mapping:
                        tracker.user_id = user_mapping[0]

                return tracker
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
                latest session (at or after the latest ``action_session_start``
                timestamp).

        Returns:
            Query to get the conversation events.
        """
        # Subquery: timestamp of the latest ``action_session_start`` action
        # (aligns with :func:`get_latest_replay_safe_session_tracker` boundary).
        session_start_sub_query = (
            session.query(sa.func.max(self.SQLEvent.timestamp).label("session_start"))
            .filter(
                self.SQLEvent.sender_id == sender_id,
                self.SQLEvent.type_name == ActionExecuted.type_name,
                self.SQLEvent.action_name == ACTION_SESSION_START_NAME,
            )
            .subquery()
        )

        event_query = session.query(self.SQLEvent).filter(
            self.SQLEvent.sender_id == sender_id
        )
        if not fetch_events_from_all_sessions:
            event_query = event_query.filter(
                # Events at or after the latest ``action_session_start``, or all events
                # if none exist.
                sa.or_(
                    self.SQLEvent.timestamp >= session_start_sub_query.c.session_start,
                    session_start_sub_query.c.session_start.is_(None),
                )
            )

        return event_query.order_by(self.SQLEvent.id)

    def _event_to_sql_event_fields(self, event: Event) -> Dict[str, Any]:
        """Build dict of SQLEvent fields from an event.

        Keys: type_name, timestamp, intent_name, action_name, data.
        """
        data = event.as_dict()
        intent = data.get("parse_data", {}).get("intent", {}).get(INTENT_NAME_KEY)
        action = data.get("name")
        timestamp = data.get("timestamp")
        return {
            "type_name": event.type_name,
            "timestamp": timestamp,
            "intent_name": intent,
            "action_name": action,
            "data": json.dumps(data),
        }

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Update database with events from the current conversation."""
        await self.stream_events(tracker)

        # Ensure conversation_started_timestamp is set (for backward compatibility)
        tracker.ensure_conversation_started_timestamp()

        with self.session_scope() as session:
            # only store recent events
            events = self._additional_events(session, tracker)

            for event in events:
                fields = self._event_to_sql_event_fields(event)

                # noinspection PyArgumentList
                session.add(
                    self.SQLEvent(
                        sender_id=tracker.sender_id,
                        **fields,
                    )
                )

            # Maintain users table if tracker has user_id
            if tracker.user_id:
                self._upsert_user_mapping(
                    session,
                    tracker.sender_id,
                    tracker.user_id,
                    tracker.conversation_started_timestamp,
                )

            session.commit()

        structlogger.debug(
            "sql_tracker_store.save_tracker",
            event_info=(
                f"Tracker with sender_id '{tracker.sender_id}' stored to database",
            ),
        )

    def _build_stored_tracker(
        self,
        session: "Session",
        sender_id: str,
        serialised_rows: list,
    ) -> DialogueStateTracker:
        """Reconstruct a :class:`DialogueStateTracker` from raw SQL event rows.

        JSON-decodes each row, builds the tracker via
        :meth:`~rasa.shared.core.trackers.DialogueStateTracker.from_dict`, and
        attaches any ``user_id`` found in the users table.

        Args:
            session: Active database session (used for the users-table lookup).
            sender_id: Conversation sender ID.
            serialised_rows: Raw :class:`SQLEvent` ORM rows whose ``.data`` field
                contains the JSON-serialised event.

        Returns:
            Reconstructed tracker, with ``user_id`` set if a mapping exists.
        """
        events = [json.loads(row.data) for row in serialised_rows]
        stored_tracker = DialogueStateTracker.from_dict(
            sender_id, events, self.domain.slots
        )
        if self._users_table_exists:
            user_mapping = (
                session.query(self.SQLUser.user_id)
                .filter(self.SQLUser.sender_id == sender_id)
                .one_or_none()
            )
            if user_mapping:
                stored_tracker.user_id = user_mapping[0]
        return stored_tracker

    def _additional_events(
        self, session: "Session", tracker: DialogueStateTracker
    ) -> Iterator:
        """Return events from the tracker which aren't currently stored.

        The offset into ``tracker.events`` matches the length of the tracker
        returned by :meth:`retrieve` (replay-safe latest session, using the
        ``action_session_start`` boundary), not the raw SQL row count for the
        timestamp filter (which can differ when timestamps collide).

        Uses a **two-phase fetch** to minimise DB I/O:

        * **Phase 1 (fast path):** fetches only the latest session's events
          (``fetch_events_from_all_sessions=False``). Sufficient for the vast
          majority of conversations.
        * **Phase 2 (fallback):** triggered only when
          :func:`~rasa.shared.core.trackers.get_latest_replay_safe_session_tracker`
          exhausts all reconstruction candidates with the session-only view —
          which happens when a second session holds ``DialogueStackUpdated`` events
          that reference frames added in an earlier session. Re-fetches full
          history and repeats reconstruction.

        The ``domain.is_empty()`` early-exit path still uses the full-history fetch
        because it needs the raw persisted-row count, not a replay-safe slice.
        """
        # --- Early exit: domain-less operation ---
        # Needs the raw count of all persisted events so that newly arrived events
        # are appended rather than re-written (no replay-safe slicing here).
        if self.domain.is_empty():
            all_rows = self._event_query(
                session, tracker.sender_id, fetch_events_from_all_sessions=True
            ).all()
            offset = len(all_rows)
            return itertools.islice(tracker.events, offset, len(tracker.events))

        # --- Phase 1: session-only fetch (fast path) ---
        session_rows = self._event_query(
            session, tracker.sender_id, fetch_events_from_all_sessions=False
        ).all()

        if not session_rows:
            # No events stored at all — this is the first save.
            return itertools.islice(tracker.events, 0, len(tracker.events))

        start_session_after_expiry = (
            self.domain.session_config.start_session_after_expiry
        )
        try:
            stored_tracker = self._build_stored_tracker(
                session, tracker.sender_id, session_rows
            )
            sliced = get_latest_replay_safe_session_tracker(
                stored_tracker, start_session_after_expiry=start_session_after_expiry
            )
        except (JsonPatchException, JsonPointerException):
            # The session-only view triggered an unrecoverable patch failure during
            # tracker reconstruction (e.g. a second session replaces a frame that was
            # only added in the first session). Fall through to Phase 2.
            sliced = None
            stored_tracker = None

        # Detect Phase 1 success:
        # - sliced is not None (no exception) AND
        # - sliced is a newly-constructed tracker (not the same object as
        #   stored_tracker, which is returned by
        #   get_latest_replay_safe_session_tracker only when every
        #   reconstruction candidate failed).
        if sliced is not None and sliced is not stored_tracker:
            return itertools.islice(
                tracker.events, len(sliced.events), len(tracker.events)
            )

        # --- Phase 2: full-history fallback ---
        # Session-only view insufficient (cross-session dialogue-stack dependencies).
        structlogger.debug(
            "sql_tracker_store._additional_events.widening_to_full_history",
            sender_id=tracker.sender_id,
            event_info=(
                "Session-only fetch insufficient for replay-safe slicing; "
                "falling back to full history fetch."
            ),
        )
        all_rows = self._event_query(
            session, tracker.sender_id, fetch_events_from_all_sessions=True
        ).all()
        stored_tracker_full = self._build_stored_tracker(
            session, tracker.sender_id, all_rows
        )
        sliced_full = get_latest_replay_safe_session_tracker(
            stored_tracker_full,
            start_session_after_expiry=start_session_after_expiry,
        )
        return itertools.islice(
            tracker.events, len(sliced_full.events), len(tracker.events)
        )

    def _upsert_user_mapping(
        self,
        session: "Session",
        sender_id: str,
        user_id: str,
        conversation_started_timestamp: Optional[float] = None,
    ) -> None:
        """Upsert user mapping in the users table.

        Args:
            session: Database session.
            sender_id: Sender ID to map.
            user_id: User ID to map to sender_id.
            conversation_started_timestamp: Optional timestamp of the first event.
        """
        # Use database-specific upsert syntax
        dialect_name = self.engine.dialect.name

        if dialect_name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stmt: Any = pg_insert(self.SQLUser).values(
                sender_id=sender_id,
                user_id=user_id,
                conversation_started_timestamp=conversation_started_timestamp,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["sender_id"],
                set_={
                    USER_ID: stmt.excluded.user_id,
                    "conversation_started_timestamp": (
                        stmt.excluded.conversation_started_timestamp
                    ),
                },
            )
            session.execute(stmt)
        elif dialect_name == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            stmt = sqlite_insert(self.SQLUser).values(
                sender_id=sender_id,
                user_id=user_id,
                conversation_started_timestamp=conversation_started_timestamp,
            )
            stmt = stmt.on_conflict_do_update(
                set_={
                    USER_ID: stmt.excluded.user_id,
                    "conversation_started_timestamp": (
                        stmt.excluded.conversation_started_timestamp
                    ),
                }
            )
            session.execute(stmt)
        else:
            self._generic_upsert(
                session,
                sender_id,
                user_id,
                conversation_started_timestamp,
            )

    def _update_events_in_place(
        self,
        session: "Session",
        existing_rows: List[Any],
        tracker_to_keep: DialogueStateTracker,
    ) -> int:
        """Update stored event rows in place where content differs.

        Returns count of rows updated.
        """
        updates_count = 0
        for db_row, new_event in zip(existing_rows, tracker_to_keep.events):
            new_fields = self._event_to_sql_event_fields(new_event)
            if new_fields["data"] != db_row.data:
                session.execute(
                    sa.update(self.SQLEvent)
                    .where(self.SQLEvent.id == db_row.id)
                    .values(
                        type_name=new_fields["type_name"],
                        timestamp=new_fields["timestamp"],
                        intent_name=new_fields["intent_name"],
                        action_name=new_fields["action_name"],
                        data=new_fields["data"],
                    )
                )
                updates_count += 1
        return updates_count

    def _replace_all_tracker_events(
        self,
        session: "Session",
        tracker_to_keep: DialogueStateTracker,
    ) -> None:
        """Delete all events for sender_id and insert events from tracker_to_keep."""
        session.execute(
            sa.delete(self.SQLEvent).where(
                self.SQLEvent.sender_id == tracker_to_keep.sender_id
            )
        )
        for event in tracker_to_keep.events:
            fields = self._event_to_sql_event_fields(event)
            # noinspection PyArgumentList
            session.add(
                self.SQLEvent(
                    sender_id=tracker_to_keep.sender_id,
                    **fields,
                )
            )

    async def update(
        self,
        tracker_to_keep: DialogueStateTracker,
        apply_deletion_only: bool = True,
    ) -> None:
        """Overwrite or trim the tracker in the SQL tracker store.

        Two intended use cases, selected by apply_deletion_only:

        1. **Trim (retention/deletion)** — apply_deletion_only=True (default): Only the
           timestamp-based delete (and user mapping) is applied. Use for deletion cron.
           When the delete removes no rows (e.g. events share the threshold timestamp),
           we do not run the content-only path, avoiding incorrect in-place updates.

        2. **Content-only (e.g. anonymization)** — apply_deletion_only=False: When the
           delete removes no rows we also apply content-only updates (in-place or full
           replace). Use for anonymization where the same event set has modified
           content.

        Args:
            tracker_to_keep: The tracker to keep (trim older events or overwrite
                content).
            apply_deletion_only: If True (default), only the delete step runs. If False,
                content-only path runs when rowcount == 0.
        """
        # Ensure conversation_started_timestamp is set (for backward compatibility)
        tracker_to_keep.ensure_conversation_started_timestamp()

        content_only_log: Optional[str] = None

        with self.session_scope() as session:
            # Delete events whose timestamp are older
            # than the first event of the tracker to keep.
            first_ts = (
                tracker_to_keep.events[0].timestamp if tracker_to_keep.events else 0.0
            )
            statement = sa.delete(self.SQLEvent).where(
                self.SQLEvent.sender_id == tracker_to_keep.sender_id,
                self.SQLEvent.timestamp < first_ts,
            )

            result = session.execute(statement)

            if not isinstance(result, sa.engine.cursor.CursorResult):
                result = cast(sa.engine.cursor.CursorResult, result)

            # Maintain users table if tracker has user_id
            if tracker_to_keep.user_id:
                self._upsert_user_mapping(
                    session,
                    tracker_to_keep.sender_id,
                    tracker_to_keep.user_id,
                    tracker_to_keep.conversation_started_timestamp,
                )

            # Content-only path: only when caller opts in (e.g. anonymization) and
            # delete removed no rows. When apply_deletion_only is True (deletion cron)
            # we never run this, even if rowcount == 0 (e.g. events share threshold ts).
            if (
                result.rowcount == 0
                and tracker_to_keep.events
                and not apply_deletion_only
            ):
                existing_rows = self._event_query(
                    session,
                    tracker_to_keep.sender_id,
                    fetch_events_from_all_sessions=True,
                ).all()
                if len(existing_rows) == len(tracker_to_keep.events):
                    updates_count = self._update_events_in_place(
                        session, existing_rows, tracker_to_keep
                    )
                    content_only_log = (
                        f"0 rows removed; {updates_count} events updated in place."
                    )
                else:
                    self._replace_all_tracker_events(session, tracker_to_keep)
                    content_only_log = (
                        "0 rows removed; tracker replaced (event count mismatch)."
                    )

            session.commit()

        first_event_timestamp = ""
        if tracker_to_keep.events:
            first_event_timestamp = str(
                datetime.fromtimestamp(tracker_to_keep.events[0].timestamp)
            )
        event_info = (
            content_only_log
            if content_only_log is not None
            else f"{result.rowcount} rows removed from tracker."
        )

        structlogger.info(
            "sql_tracker_store.update.updated_tracker",
            sender_id=tracker_to_keep.sender_id,
            first_event_timestamp=first_event_timestamp,
            event_info=event_info,
        )

    async def get_trackers_by_user_id(
        self,
        user_id: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[DialogueStateTracker]:
        """Retrieves all trackers for a given user_id using efficient JOIN query.

        This method uses the users table for efficient querying.
        If the users table doesn't exist or is empty, it logs a warning and
        returns an empty list.

        Args:
            user_id: User ID to fetch trackers for.
            limit: Optional maximum number of trackers to return. If None, returns all
                matching trackers.
            skip: Optional number of trackers to skip before returning results. If None,
                starts from the beginning.

        Returns:
            List of trackers associated with the user_id.
        """
        if not self._users_table_exists:
            structlogger.warning(
                "sql_tracker_store.get_trackers_by_user_id.no_users_table",
                event_info=(
                    "Users table does not exist. To enable efficient "
                    "querying by user_id, please ensure the users "
                    "table is created by using a recent version of Rasa and that "
                    "trackers are saved with user_id set."
                ),
            )
            return []

        # Use efficient JOIN query with users table
        with self.session_scope() as session:
            # Query sender_ids with conversation_started_timestamp for efficient
            # sorting. Sort by conversation_started_timestamp (if available) then
            # sender_id at database level. Use COALESCE to handle NULL values
            # (treat NULL as 0.0 for sorting)
            query = (
                session.query(
                    self.SQLUser.sender_id,
                )
                .filter(self.SQLUser.user_id == user_id)
                .order_by(
                    # Sort by conversation_started_timestamp (NULLS treated as
                    # 0.0), then sender_id
                    sa.func.coalesce(
                        self.SQLUser.conversation_started_timestamp, 0.0
                    ).asc(),
                    self.SQLUser.sender_id.asc(),
                )
            )

            # Apply pagination at database level for efficiency
            if skip is not None and skip > 0:
                query = query.offset(skip)
            if limit is not None and limit > 0:
                query = query.limit(limit)

            sender_ids = [row[0] for row in query.all()]

        # Retrieve trackers for matching sender_ids
        trackers = []
        for sender_id in sender_ids:
            tracker = await self.retrieve_full_tracker(sender_id)
            if tracker is not None:
                trackers.append(tracker)

        return trackers

    def _generic_upsert(
        self,
        session: "Session",
        sender_id: str,
        user_id: str,
        conversation_started_timestamp: Optional[float] = None,
    ) -> None:
        """Use UPDATE expression followed by INSERT if needed.

        This is required for SQL databases other than PostgreSQL and SQLite.
        Implement try-except to handle race conditions.
        Use a savepoint to isolate the upsert operation so rollback doesn't
        affect other pending changes (e.g., events) in the transaction.
        """
        savepoint = session.begin_nested()

        try:
            # Attempt to update existing row first
            update_stmt = (
                sa.update(self.SQLUser)
                .where(self.SQLUser.sender_id == sender_id)
                .values(
                    user_id=user_id,
                    conversation_started_timestamp=conversation_started_timestamp,
                )
            )
            result = session.execute(update_stmt)

            if not isinstance(result, sa.engine.cursor.CursorResult):
                result = cast(sa.engine.cursor.CursorResult, result)

            # If no rows were updated, the row doesn't exist - insert it
            if result.rowcount == 0:
                session.add(
                    self.SQLUser(
                        sender_id=sender_id,
                        user_id=user_id,
                        conversation_started_timestamp=conversation_started_timestamp,
                    )
                )

            # Commit the savepoint - this will flush changes within the savepoint
            # and raise IntegrityError if a constraint violation occurs
            savepoint.commit()
        except sa.exc.IntegrityError:
            # In case of race condition, rollback only the savepoint, not the
            # entire transaction, so events and other pending changes are preserved.
            savepoint.rollback()
            # The row already exists with the correct user_id, so we can
            # continue with the main transaction
