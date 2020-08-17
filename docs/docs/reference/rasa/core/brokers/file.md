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

#### publish

```python
 | publish(event: Dict) -> None
```

Write event to file.

