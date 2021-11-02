---
sidebar_label: rasa.core.evaluation.marker
title: rasa.core.evaluation.marker
---
## AndMarker Objects

```python
@MarkerRegistry.configurable_marker
class AndMarker(OperatorMarker)
```

Checks that all sub-markers apply.

#### positive\_tag

```python
 | @staticmethod
 | positive_tag() -> Text
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
class OrMarker(OperatorMarker)
```

Checks that at least one sub-marker applies.

#### positive\_tag

```python
 | @staticmethod
 | positive_tag() -> Text
```

Returns the tag to be used in a config file.

## NotMarker Objects

```python
@MarkerRegistry.configurable_marker
class NotMarker(OperatorMarker)
```

Checks that at least one sub-marker applies.

#### positive\_tag

```python
 | @staticmethod
 | positive_tag() -> Text
```

Returns the tag to be used in a config file.

#### expected\_number\_of\_sub\_markers

```python
 | @staticmethod
 | expected_number_of_sub_markers() -> Optional[int]
```

Returns the expected number of sub-markers (if there is any).

## SequenceMarker Objects

```python
@MarkerRegistry.configurable_marker
class SequenceMarker(OperatorMarker)
```

Checks that all sub-markers apply consecutively in the specified order.

Given a sequence of sub-markers `m_0, m_1,...,m_n`, the sequence marker applies
at the `i`-th event if sub-marker `m_{n-j}` applies at the `{i-j}`-th event
for `j` in `[0,..,n]`.

#### positive\_tag

```python
 | @staticmethod
 | positive_tag() -> Text
```

Returns the tag to be used in a config file.

#### negated\_tag

```python
 | @staticmethod
 | negated_tag() -> Text
```

Returns the tag to be used in a config file.

## OccurrenceMarker Objects

```python
@MarkerRegistry.configurable_marker
class OccurrenceMarker(OperatorMarker)
```

Checks that all sub-markers applied at least once in history.

It doesn&#x27;t matter if the sub markers stop applying later in history. If they
applied at least once they will always evaluate to `True`.

#### positive\_tag

```python
 | @staticmethod
 | positive_tag() -> Text
```

Returns the tag to be used in a config file.

#### negated\_tag

```python
 | @staticmethod
 | negated_tag() -> Optional[Text]
```

Returns the tag to be used in a config file for the negated version.

#### expected\_number\_of\_sub\_markers

```python
 | @staticmethod
 | expected_number_of_sub_markers() -> Optional[int]
```

Returns the expected number of sub-markers (if there is any).

#### relevant\_events

```python
 | relevant_events() -> List[int]
```

Only return index of first match (see parent class for full docstring).

## ActionExecutedMarker Objects

```python
@MarkerRegistry.configurable_marker
class ActionExecutedMarker(ConditionMarker)
```

Checks whether an action is executed at the current step.

#### positive\_tag

```python
 | @staticmethod
 | positive_tag() -> Text
```

Returns the tag to be used in a config file.

#### negated\_tag

```python
 | @staticmethod
 | negated_tag() -> Optional[Text]
```

Returns the tag to be used in a config file for the negated version.

#### validate\_against\_domain

```python
 | validate_against_domain(domain: Domain) -> bool
```

Checks that this marker (and its children) refer to entries in the domain.

**Arguments**:

- `domain` - The domain to check against

## IntentDetectedMarker Objects

```python
@MarkerRegistry.configurable_marker
class IntentDetectedMarker(ConditionMarker)
```

Checks whether an intent is expressed at the current step.

More precisely it applies at an event if this event is a `UserUttered` event
where either (1) the retrieval intent or (2) just the intent coincides with
the specified text.

#### positive\_tag

```python
 | @staticmethod
 | positive_tag() -> Text
```

Returns the tag to be used in a config file.

#### negated\_tag

```python
 | @staticmethod
 | negated_tag() -> Optional[Text]
```

Returns the tag to be used in a config file for the negated version.

#### validate\_against\_domain

```python
 | validate_against_domain(domain: Domain) -> bool
```

Checks that this marker (and its children) refer to entries in the domain.

**Arguments**:

- `domain` - The domain to check against

## SlotSetMarker Objects

```python
@MarkerRegistry.configurable_marker
class SlotSetMarker(ConditionMarker)
```

Checks whether a slot is set at the current step.

The actual `SlotSet` event might have happened at an earlier step.

#### positive\_tag

```python
 | @staticmethod
 | positive_tag() -> Text
```

Returns the tag to be used in a config file.

#### negated\_tag

```python
 | @staticmethod
 | negated_tag() -> Optional[Text]
```

Returns the tag to be used in a config file for the negated version.

#### validate\_against\_domain

```python
 | validate_against_domain(domain: Domain) -> bool
```

Checks that this marker (and its children) refer to entries in the domain.

**Arguments**:

- `domain` - The domain to check against.

