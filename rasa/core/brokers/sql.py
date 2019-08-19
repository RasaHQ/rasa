import json
import logging
from typing import Any, Dict, Optional, Text

from rasa.core.brokers.event_channel import EventChannel
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


class SQLProducer(EventChannel):
    """Save events into an SQL database.

    All events will be stored in a table called `events`.

    """

    from sqlalchemy.ext.declarative import declarative_base

    Base = declarative_base()

    class SQLBrokerEvent(Base):
        from sqlalchemy import Column, Integer, String, Text

        __tablename__ = "events"
        id = Column(Integer, primary_key=True)
        sender_id = Column(String(255))
        data = Column(Text)

    def __init__(
        self,
        dialect: Text = "sqlite",
        host: Optional[Text] = None,
        port: Optional[int] = None,
        db: Text = "events.db",
        username: Optional[Text] = None,
        password: Optional[Text] = None,
    ):
        from rasa.core.tracker_store import SQLTrackerStore
        import sqlalchemy.orm

        engine_url = SQLTrackerStore.get_db_url(
            dialect, host, port, db, username, password
        )

        logger.debug("SQLProducer: Connecting to database: '{}'.".format(engine_url))

        self.engine = sqlalchemy.create_engine(engine_url)
        self.Base.metadata.create_all(self.engine)
        self.session = sqlalchemy.orm.sessionmaker(bind=self.engine)()

    @classmethod
    def from_endpoint_config(cls, broker_config: EndpointConfig) -> "EventChannel":
        return cls(host=broker_config.url, **broker_config.kwargs)

    def publish(self, event: Dict[Text, Any]) -> None:
        """Publishes a json-formatted Rasa Core event into an event queue."""
        self.session.add(
            self.SQLBrokerEvent(
                sender_id=event.get("sender_id"), data=json.dumps(event)
            )
        )
        self.session.commit()
