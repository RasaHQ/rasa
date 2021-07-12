---
sidebar_label: rasa.core.brokers.sql
title: rasa.core.brokers.sql
---
## SQLEventBroker Objects

```python
class SQLEventBroker(EventBroker)
```

Save events into an SQL database.

All events will be stored in a table called `events`.

## SQLBrokerEvent Objects

```python
class SQLBrokerEvent(Base)
```

ORM which represents a row in the `events` table.

#### from\_endpoint\_config

```python
 | @classmethod
 | async from_endpoint_config(cls, broker_config: EndpointConfig, event_loop: Optional[AbstractEventLoop] = None) -> "SQLEventBroker"
```

Creates broker. See the parent class for more information.

#### session\_scope

```python
 | @contextlib.contextmanager
 | session_scope() -> Generator[Session, None, None]
```

Provide a transactional scope around a series of operations.

#### publish

```python
 | publish(event: Dict[Text, Any]) -> None
```

Publishes a json-formatted Rasa Core event into an event queue.

