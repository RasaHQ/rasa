---
sidebar_label: rasa.core.evaluation.marker
title: rasa.core.evaluation.marker
---
## AndMarker Objects

```python
@MarkerRegistry.configurable_marker
class AndMarker(CompoundMarker)
```

Checks that all sub-markers apply.

#### tag

```python
 | @staticmethod
 | tag() -> Text
```

Returns the tag to be used in a config file.

#### negated\_tag

```python
 | @staticmethod
 | negated_tag() -> Text
```

Returns the tag to be used in a config file for the negated version.

## OrMarker Objects

```python
@MarkerRegistry.configurable_marker
class OrMarker(CompoundMarker)
```

Checks that at least one sub-marker applies.

#### tag

```python
 | @staticmethod
 | tag() -> Text
```

Returns the tag to be used in a config file.

#### negated\_tag

```python
 | @staticmethod
 | negated_tag() -> Optional[Text]
```

Returns the tag to be used in a config file for the negated version.

## SequenceMarker Objects

```python
@MarkerRegistry.configurable_marker
class SequenceMarker(CompoundMarker)
```

Checks that all sub-markers apply consecutively in the specified order.

Given a sequence of sub-markers `m_0, m_1,...,m_n`, the sequence marker applies
at the `i`-th event if sub-marker `m_{n-j}` applies at the `{i-j}`-th event
for `j` in `[0,..,n]`.

#### tag

```python
 | @staticmethod
 | tag() -> Text
```

Returns the tag to be used in a config file.

## OccurrenceMarker Objects

```python
@MarkerRegistry.configurable_marker
class OccurrenceMarker(CompoundMarker)
```

Checks that all sub-markers applied at least once in history.

It doesn&#x27;t matter if the sub markers stop applying later in history. If they
applied at least once they will always evaluate to `True`.

#### \_\_init\_\_

```python
 | __init__(markers: List[Marker], negated: bool = False, name: Optional[Text] = None) -> None
```

Creates marker (see parent class for full docstring).

#### tag

```python
 | @staticmethod
 | tag() -> Text
```

Returns the tag to be used in a config file.

#### negated\_tag

```python
 | @staticmethod
 | negated_tag() -> Optional[Text]
```

Returns the tag to be used in a config file for the negated version.

#### relevant\_events

```python
 | relevant_events() -> List[int]
```

Only return index of first match (see parent class for full docstring).

## ActionExecutedMarker Objects

```python
@MarkerRegistry.configurable_marker
class ActionExecutedMarker(AtomicMarker)
```

Checks whether an action is executed at the current step.

#### tag

```python
 | @staticmethod
 | tag() -> Text
```

Returns the tag to be used in a config file.

#### negated\_tag

```python
 | @staticmethod
 | negated_tag() -> Optional[Text]
```

Returns the tag to be used in a config file for the negated version.

## IntentDetectedMarker Objects

```python
@MarkerRegistry.configurable_marker
class IntentDetectedMarker(AtomicMarker)
```

Checks whether an intent is expressed at the current step.

More precisely it applies at an event if this event is a `UserUttered` event
where either (1) the retrieval intent or (2) just the intent coincides with
the specified text.

#### tag

```python
 | @staticmethod
 | tag() -> Text
```

Returns the tag to be used in a config file.

#### negated\_tag

```python
 | @staticmethod
 | negated_tag() -> Optional[Text]
```

Returns the tag to be used in a config file for the negated version.

## SlotSetMarker Objects

```python
@MarkerRegistry.configurable_marker
class SlotSetMarker(AtomicMarker)
```

Checks whether a slot is set at the current step.

The actual `SlotSet` event might have happened at an earlier step.

#### tag

```python
 | @staticmethod
 | tag() -> Text
```

Returns the tag to be used in a config file.

#### negated\_tag

```python
 | @staticmethod
 | negated_tag() -> Optional[Text]
```

Returns the tag to be used in a config file for the negated version.

