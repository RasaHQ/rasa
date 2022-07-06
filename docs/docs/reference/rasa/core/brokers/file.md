---
sidebar_label: rasa.core.brokers.file
title: rasa.core.brokers.file
---
## FileEventBroker Objects

```python
class FileEventBroker(EventBroker)
```

Log events to a file in json format.

There will be one event per line and each event is stored as json.

#### from\_endpoint\_config

```python
@classmethod
async def from_endpoint_config(cls, broker_config: Optional["EndpointConfig"], event_loop: Optional[AbstractEventLoop] = None) -> Optional["FileEventBroker"]
```

Creates broker. See the parent class for more information.

#### publish

```python
def publish(event: Dict) -> None
```

Write event to file.

