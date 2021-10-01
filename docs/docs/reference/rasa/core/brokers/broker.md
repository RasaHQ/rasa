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
@staticmethod
async def create(obj: Union["EventBroker", EndpointConfig, None], loop: Optional[AbstractEventLoop] = None) -> Optional["EventBroker"]
```

Factory to create an event broker.

#### from\_endpoint\_config

```python
@classmethod
async def from_endpoint_config(cls, broker_config: EndpointConfig, event_loop: Optional[AbstractEventLoop] = None) -> "EventBroker"
```

Creates an `EventBroker` from the endpoint configuration.

**Arguments**:

- `broker_config` - The configuration for the broker.
- `event_loop` - The current event loop or `None`.
  

**Returns**:

  An `EventBroker` object.

#### publish

```python
def publish(event: Dict[Text, Any]) -> None
```

Publishes a json-formatted Rasa Core event into an event queue.

#### is\_ready

```python
def is_ready() -> bool
```

Determine whether or not the event broker is ready.

**Returns**:

  `True` by default, but this may be overridden by subclasses.

#### close

```python
async def close() -> None
```

Close the connection to an event broker.

