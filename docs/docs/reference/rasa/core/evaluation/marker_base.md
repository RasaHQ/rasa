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

## EventMetaData Objects

```python
@dataclass
class EventMetaData()
```

Describes meta data per event in some session.

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

**Raises**:

  `InvalidMarkerConfig` if the chosen *name* of the marker is the tag of
  a predefined marker.

#### get\_tag

```python
 | get_tag() -> Text
```

Returns the tag describing this marker.

#### positive\_tag

```python
 | @staticmethod
 | @abstractmethod
 | positive_tag() -> Text
```

Returns the tag to be used in a config file.

#### negated\_tag

```python
 | @staticmethod
 | negated_tag() -> Optional[Text]
```

Returns the tag to be used in a config file for the negated version.

If this maps to `None`, then this indicates that we do not allow a short-cut
for negating this marker. Hence, there is not a single tag to instantiate
a negated version of this marker. One must use a &quot;not&quot; in the configuration
file then.

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

#### \_\_len\_\_

```python
 | @abstractmethod
 | __len__() -> int
```

Returns the count of all markers that are part of this marker.

#### validate\_against\_domain

```python
 | @abstractmethod
 | validate_against_domain(domain: Domain) -> bool
```

Checks that this marker (and its children) refer to entries in the domain.

**Arguments**:

- `domain` - The domain to check against

#### evaluate\_events

```python
 | evaluate_events(events: List[Event], recursive: bool = False) -> List[Dict[Text, List[EventMetaData]]]
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

  a list that contains, for each session contained in the tracker, a
  dictionary mapping that maps marker names to meta data of relevant
  events

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
 | @classmethod
 | from_path(cls, path: Union[Path, Text]) -> Marker
```

Loads markers from one config file or all config files in a directory tree.

Each config file should contain a dictionary mapping marker names to the
respective marker configuration.
To avoid confusion, the marker names must not coincide with the tag of
any pre-defined markers. Moreover, marker names must be unique. This means,
if you you load the markers from multiple files, then you have to make sure
that the names of the markers defined in these files are unique across all
loaded files.

Note that all loaded markers will be combined into one marker via one
artificial OR-operator. When evaluating the resulting marker, then the
artificial OR-operator will be ignored and results will be reported for
every sub-marker.

For more details how a single marker configuration looks like, have a look
at `Marker.from_config`.

**Arguments**:

- `path` - either the path to a single config file or the root of the directory
  tree that contains marker config files

**Returns**:

  all configured markers, combined into one marker object

#### from\_config

```python
 | @staticmethod
 | from_config(config: Any, name: Optional[Text] = None) -> Marker
```

Creates a marker from the given config.

A marker configuration is a dictionary mapping a marker tag (either a
`positive_tag` or a `negated_tag`) to a sub-configuration.
How that sub-configuration looks like, depends on whether the tag describes
an operator (see `OperatorMarker.from_tag_and_sub_config`) or a
condition (see `ConditionMarker.from_tag_and_sub_config`).

**Arguments**:

- `config` - a marker configuration
- `name` - a custom name that will be used for the top-level marker (if and
  only if there is only one top-level marker)
  

**Returns**:

  the configured marker

#### export\_markers

```python
 | export_markers(tracker_loader: Iterator[Optional[DialogueStateTracker]], output_file: Text, stats_file: Optional[Text] = None) -> None
```

Collect markers for each dialogue in each tracker loaded.

**Arguments**:

- `tracker_loader` - The tracker loader to use to select trackers for marker
  extraction.
- `output_file` - Path to write out the extracted markers.
- `stats_file` - (Optional) Path to write out statistics about the extracted
  markers.

## OperatorMarker Objects

```python
class OperatorMarker(Marker,  ABC)
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

**Raises**:

  `InvalidMarkerConfig` if the given number of sub-markers does not match
  the expected number of sub-markers

#### expected\_number\_of\_sub\_markers

```python
 | @staticmethod
 | expected_number_of_sub_markers() -> Optional[int]
```

Returns the expected number of sub-markers (if there is any).

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

#### \_\_len\_\_

```python
 | __len__() -> int
```

Returns the count of all markers that are part of this marker.

#### reset

```python
 | reset() -> None
```

Resets the history of this marker and all its sub-markers.

#### validate\_against\_domain

```python
 | validate_against_domain(domain: Domain) -> bool
```

Checks that this marker (and its children) refer to entries in the domain.

**Arguments**:

- `domain` - The domain to check against

#### from\_tag\_and\_sub\_config

```python
 | @staticmethod
 | from_tag_and_sub_config(tag: Text, sub_config: Any, name: Optional[Text] = None) -> OperatorMarker
```

Creates an operator marker from the given config.

The configuration must consist of a list of marker configurations.
See `Marker.from_config` for more details.

**Arguments**:

- `tag` - the tag identifying an operator
- `sub_config` - a list of marker configs
- `name` - an optional custom name to be attached to the resulting marker

**Returns**:

  the configured operator marker

**Raises**:

  `InvalidMarkerConfig` if the given config or the tag are not well-defined

## ConditionMarker Objects

```python
class ConditionMarker(Marker,  ABC)
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
 | __iter__() -> Iterator[ConditionMarker]
```

Returns an iterator that just returns this `AtomicMarker`.

**Returns**:

  an iterator over all markers that are part of this marker, i.e. this marker

#### \_\_len\_\_

```python
 | __len__() -> int
```

Returns the count of all markers that are part of this marker.

#### from\_tag\_and\_sub\_config

```python
 | @staticmethod
 | from_tag_and_sub_config(tag: Text, sub_config: Any, name: Optional[Text] = None) -> ConditionMarker
```

Creates an atomic marker from the given config.

**Arguments**:

- `tag` - the tag identifying a condition
- `sub_config` - a single text parameter expected by all condition markers;
  e.g. if the tag is for an `intent_detected` marker then the `config`
  should contain an intent name
- `name` - a custom name for this marker

**Returns**:

  the configured `ConditionMarker`

**Raises**:

  `InvalidMarkerConfig` if the given config or the tag are not well-defined

