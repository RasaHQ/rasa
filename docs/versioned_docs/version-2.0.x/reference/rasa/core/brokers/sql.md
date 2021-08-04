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

#### session\_scope

```python
 | @contextlib.contextmanager
 | session_scope()
```

Provide a transactional scope around a series of operations.

#### publish

```python
 | publish(event: Dict[Text, Any]) -> None
```

Publishes a json-formatted Rasa Core event into an event queue.

