---
sidebar_label: rasa.core.exporter
title: rasa.core.exporter
---
## Exporter Objects

```python
class Exporter()
```

Manages the publishing of events in a tracker store to an event broker.

**Attributes**:

- `endpoints_path` - Path to the endpoints file used to configure the event
  broker and tracker store. If `None`, the default path (&#x27;endpoints.yml&#x27;)
  is used.
- `tracker_store` - `TrackerStore` to export conversations from.
- `event_broker` - `EventBroker` to export conversations to.
- `requested_conversation_ids` - List of conversation IDs requested to be
  processed.
- `minimum_timestamp` - Minimum timestamp of events that are published.
  If `None`, apply no such constraint.
- `maximum_timestamp` - Maximum timestamp of events that are published.
  If `None`, apply no such constraint.

#### publish\_events

```python
 | async publish_events() -> int
```

Publish events in a tracker store using an event broker.

Exits if the publishing of events is interrupted due to an error. In that case,
the CLI command to continue the export where it was interrupted is printed.

**Returns**:

  The number of successfully published events.

