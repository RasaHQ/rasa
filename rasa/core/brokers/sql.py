import contextlib
import json
import logging
from asyncio import AbstractEventLoop
from typing import Any, Dict, Optional, Text, Generator

from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy import Column, Integer, String
from sqlalchemy import Text as SqlAlchemyText  # to avoid name clash with typing.Text

from rasa.core.brokers.broker import EventBroker
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


class SQLEventBroker(EventBroker):
    """Save events into an SQL database.

    All events will be stored in a table called `events`.

    """

    Base: DeclarativeMeta = declarative_base()

    class SQLBrokerEvent(Base):
        """ORM which represents a row in the `events` table."""

        __tablename__ = "events"
        id = Column(Integer, primary_key=True)
        sender_id = Column(String(255))
        data = Column(SqlAlchemyText)

    def __init__(
        self,
        dialect: Text = "sqlite",
        host: Optional[Text] = None,
        port: Optional[int] = None,
        db: Text = "events.db",
        username: Optional[Text] = None,
        password: Optional[Text] = None,
    ) -> None:
        """Initializes `SQLBrokerEvent`."""
        from rasa.core.tracker_store import SQLTrackerStore
        import sqlalchemy.orm

        engine_url = SQLTrackerStore.get_db_url(
            dialect, host, port, db, username, password
        )

        logger.debug(f"SQLEventBroker: Connecting to database: '{engine_url}'.")

        self.engine = sqlalchemy.create_engine(engine_url)
        self.Base.metadata.create_all(self.engine)
        self.sessionmaker = sqlalchemy.orm.sessionmaker(bind=self.engine)

    @classmethod
    async def from_endpoint_config(
        cls,
        broker_config: EndpointConfig,
        event_loop: Optional[AbstractEventLoop] = None,
    ) -> "SQLEventBroker":
        """Creates broker. See the parent class for more information."""
        return cls(host=broker_config.url, **broker_config.kwargs)

    @contextlib.contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self.sessionmaker()
        try:
            yield session
        finally:
            session.close()

    def publish(self, event: Dict[Text, Any]) -> None:
        """Publishes a json-formatted Rasa Core event into an event queue."""
        with self.session_scope() as session:
            session.add(
                self.SQLBrokerEvent(
                    sender_id=event.get("sender_id"), data=json.dumps(event)
                )
            )
            session.commit()
