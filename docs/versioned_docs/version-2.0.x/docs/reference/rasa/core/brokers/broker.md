---
sidebar_label: rasa.core.brokers.broker
title: rasa.core.brokers.broker
---

## EventBroker Objects

```python
class EventBroker()
```

Base class for any event broker implementation.

#### create

```python
 | @staticmethod
 | create(obj: Union["EventBroker", EndpointConfig, None]) -> Optional["EventBroker"]
```

Factory to create an event broker.

#### publish

```python
 | publish(event: Dict[Text, Any]) -> None
```

Publishes a json-formatted Rasa Core event into an event queue.

#### is\_ready

```python
 | is_ready() -> bool
```

Determine whether or not the event broker is ready.

**Returns**:

  `True` by default, but this may be overridden by subclasses.

#### close

```python
 | close() -> None
```

Close the connection to an event broker.

