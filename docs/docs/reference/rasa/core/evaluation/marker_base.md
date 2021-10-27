---
sidebar_label: rasa.core.evaluation.marker_base
title: rasa.core.evaluation.marker_base
---
## MarkerRegistry Objects

```python
class MarkerRegistry()
```

Keeps track of tags that can be used to configure markers.

#### register\_builtin\_markers

```python
 | @classmethod
 | register_builtin_markers(cls) -> None
```

Must import all modules containing markers.

#### configurable\_marker

```python
 | @classmethod
 | configurable_marker(cls, marker_class: Type[Marker]) -> Type[Marker]
```

Decorator used to register a marker that can be used in config files.

**Arguments**:

- `marker_class` - the marker class to be made available via config files

**Returns**:

  the registered marker class

#### get\_non\_negated\_tag

```python
 | @classmethod
 | get_non_negated_tag(cls, tag_or_negated_tag: Text) -> Tuple[Text, bool]
```

Returns the non-negated marker tag, given a (possible) negated marker tag.

**Arguments**:

- `tag_or_negated_tag` - the tag for a possibly negated marker

**Returns**:

  the tag itself if it was already positive, otherwise the positive version;
  and a boolean that represents whether the given tag was a negative one

## InvalidMarkerConfig Objects

```python
class InvalidMarkerConfig(RasaException)
```

Exception that can be raised when the config for a marker is not valid.

## DialogueMetaData Objects

```python
@dataclass
class DialogueMetaData()
```

Describes meta data per event in some dialogue.

#### filter

```python
 | filter(indices: List[int]) -> DialogueMetaData
```

Return a list containing meta data for the requested event indices.

**Arguments**:

- `indices` - indices of events for which we want to extract meta data

**Returns**:

  a new meta data object containing the entries for the requested indices

## Marker Objects

```python
class Marker(ABC)
```

A marker is a way of describing points in conversations you&#x27;re interested in.

Here, markers are stateful objects because they track the events of a conversation.
At each point in the conversation, one can observe whether a marker applies or
does not apply to the conversation so far.

#### \_\_init\_\_

```python
 | __init__(name: Optional[Text] = None, negated: bool = False) -> None
```

Instantiates a marker.

**Arguments**:

- `name` - a custom name that can be used to replace the default string
  conversion of this marker
- `negated` - whether this marker should be negated (i.e. a negated marker
  applies if and only if the non-negated marker does not apply)

#### tag

```python
 | @classmethod
 | @abstractmethod
 | tag(cls) -> Text
```

Returns the tag to be used in a config file.

#### negated\_tag

```python
 | @classmethod
 | negated_tag(cls) -> Optional[Text]
```

Returns the tag to be used in a config file for the negated version.

#### track

```python
 | track(event: Event) -> None
```

Updates the marker according to the given event.

**Arguments**:

- `event` - the next event of the conversation

#### reset

```python
 | reset() -> None
```

Clears the history of the marker.

#### \_\_iter\_\_

```python
 | @abstractmethod
 | __iter__() -> Iterator[Marker]
```

Returns an iterator over all markers that are part of this marker.

**Returns**:

  an iterator over all markers that are part of this marker

#### evaluate\_events

```python
 | evaluate_events(events: List[Event], recursive: bool = False) -> List[Dict[Text, DialogueMetaData]]
```

Resets the marker, tracks all events, and collects some information.

The collected information includes:
- the timestamp of each event where the marker applied and
- the number of user turns that preceded that timestamp

If this marker is the special `ANY_MARKER` (identified by its name), then
results will be collected for all (immediate) sub-markers.

If `recursive` is set to `True`, then all included markers are evaluated.

**Arguments**:

- `events` - a list of events describing a conversation
- `recursive` - set this to `True` to collect evaluations for all markers that
  this marker consists of

**Returns**:

  a list of evaluations containing one dictionary mapping marker names
  to dialogue meta data each dialogue contained in the tracker

#### relevant\_events

```python
 | relevant_events() -> List[int]
```

Returns the indices of those tracked events that are relevant for evaluation.

Note: Overwrite this method if you create a new marker class that should *not*
contain meta data about each event where the marker applied in the final
evaluation (see `evaluate_events`).

**Returns**:

  indices of tracked events

#### from\_path

```python
 | @staticmethod
 | from_path(path: Union[Path, Text]) -> Marker
```

Loads markers from one config file or all config files in a directory tree.

For more details, see `from_config_dict`.

**Arguments**:

- `path` - either the path to a single config file or the root of the directory
  tree that contains marker config files

**Returns**:

  all configured markers, combined into one marker object

#### from\_config\_dict

```python
 | @staticmethod
 | from_config_dict(config: Dict[Text, MarkerConfig]) -> Marker
```

Creates markers from a dictionary of marker configurations.

If there is more than one custom marker defined in the given dictionary,
then the returned marker will be an `or` combination of all defined markers
named `ANY_MARKER`.
During evaluation, where we usually only return results for the top-level
marker, we identify this special marker by it&#x27;s name and return evaluations
for all combined markers instead.

**Arguments**:

- `config` - mapping custom marker names to marker configurations

**Returns**:

  all configured markers, combined into one marker

#### from\_config

```python
 | @staticmethod
 | from_config(config: MarkerConfig, name: Optional[Text] = None) -> Marker
```

Creates a marker from the given config.

**Arguments**:

- `config` - the configuration of a single or multiple markers
- `name` - a custom name that will be used for the top-level marker (if and
  only if there is only one top-level marker)
  

**Returns**:

  the configured marker

## CompoundMarker Objects

```python
class CompoundMarker(Marker,  ABC)
```

Combines several markers into one.

#### \_\_init\_\_

```python
 | __init__(markers: List[Marker], negated: bool = False, name: Optional[Text] = None) -> None
```

Instantiates a marker.

**Arguments**:

- `markers` - the list of markers to combine
- `negated` - whether this marker should be negated (i.e. a negated marker
  applies if and only if the non-negated marker does not apply)
- `name` - a custom name that can be used to replace the default string
  conversion of this marker

#### track

```python
 | track(event: Event) -> None
```

Updates the marker according to the given event.

All sub-markers will be updated before the compound marker itself is updated.

**Arguments**:

- `event` - the next event of the conversation

#### \_\_iter\_\_

```python
 | __iter__() -> Iterator[Marker]
```

Returns an iterator over all included markers, plus this marker itself.

**Returns**:

  an iterator over all markers that are part of this marker

#### reset

```python
 | reset() -> None
```

Evaluate this marker given the next event.

**Arguments**:

- `event` - the next event of the conversation

#### from\_config

```python
 | @staticmethod
 | from_config(operator: Text, sub_marker_configs: List[Union[ConditionConfigList, OperatorConfig]], name: Optional[Text] = None) -> Marker
```

Creates a compound marker from the given config.

**Arguments**:

- `operator` - a text identifying a compound marker type
- `sub_marker_configs` - a list of configs defining sub-markers
- `name` - an optional custom name to be attached to the resulting marker

**Returns**:

  a compound marker if there are markers to combine - and just an event
  marker if there is only a single marker

## AtomicMarker Objects

```python
class AtomicMarker(Marker,  ABC)
```

A marker that does not contain any sub-markers.

#### \_\_init\_\_

```python
 | __init__(text: Text, negated: bool = False, name: Optional[Text] = None) -> None
```

Instantiates an atomic marker.

**Arguments**:

- `text` - some text used to decide whether the marker applies
- `negated` - whether this marker should be negated (i.e. a negated marker
  applies if and only if the non-negated marker does not apply)
- `name` - a custom name that can be used to replace the default string
  conversion of this marker

#### \_\_iter\_\_

```python
 | __iter__() -> Iterator[AtomicMarker]
```

Returns an iterator that just returns this `AtomicMarker`.

**Returns**:

  an iterator over all markers that are part of this marker, i.e. this marker

#### from\_config

```python
 | @staticmethod
 | from_config(marker_name: Text, sub_marker_config: Union[Text, List[Text]], name: Optional[Text] = None) -> List[AtomicMarker]
```

Creates an atomic marker from the given config.

**Arguments**:

- `marker_name` - string identifying an atomic marker type
- `sub_marker_config` - a list of texts or just one text which should be
  used to instantiate the condition marker(s)
- `name` - a custom name for this marker

**Returns**:

  the configured `AtomicMarker`s

