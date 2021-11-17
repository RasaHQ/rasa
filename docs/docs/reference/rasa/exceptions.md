---
sidebar_label: rasa.exceptions
title: rasa.exceptions
---
## UnsupportedModelVersionError Objects

```python
@dataclass
class UnsupportedModelVersionError(RasaException)
```

Raised when a model is too old to be loaded.

**Arguments**:

- `model_version` - the used model version that is not supported and triggered
  this exception

## ModelNotFound Objects

```python
class ModelNotFound(RasaException)
```

Raised when a model is not found in the path provided by the user.

## NoEventsToMigrateError Objects

```python
class NoEventsToMigrateError(RasaException)
```

Raised when no events to be migrated are found.

## NoConversationsInTrackerStoreError Objects

```python
class NoConversationsInTrackerStoreError(RasaException)
```

Raised when a tracker store does not contain any conversations.

## NoEventsInTimeRangeError Objects

```python
class NoEventsInTimeRangeError(RasaException)
```

Raised when a tracker store does not contain events within a given time range.

## MissingDependencyException Objects

```python
class MissingDependencyException(RasaException)
```

Raised if a python package dependency is needed, but not installed.

## PublishingError Objects

```python
@dataclass
class PublishingError(RasaException)
```

Raised when publishing of an event fails.

**Attributes**:

- `timestamp` - Unix timestamp of the event during which publishing fails.

#### \_\_str\_\_

```python
 | __str__() -> Text
```

Returns string representation of exception.

## ActionLimitReached Objects

```python
class ActionLimitReached(RasaException)
```

Raised when predicted action limit is reached.

