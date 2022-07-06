---
sidebar_label: rasa.core.training.story_conflict
title: rasa.core.training.story_conflict
---
## StoryConflict Objects

```python
class StoryConflict()
```

Represents a conflict between two or more stories.

Here, a conflict means that different actions are supposed to follow from
the same dialogue state, which most policies cannot learn.

#### \_\_init\_\_

```python
def __init__(sliced_states: List[State]) -> None
```

Creates a `StoryConflict` from a given state.

**Arguments**:

- `sliced_states` - The (sliced) dialogue state at which the conflict occurs.

#### add\_conflicting\_action

```python
def add_conflicting_action(action: Text, story_name: Text) -> None
```

Adds another action that follows from the same state.

**Arguments**:

- `action` - Name of the action.
- `story_name` - Name of the story where this action is chosen.

#### conflicting\_actions

```python
@property
def conflicting_actions() -> List[Text]
```

List of conflicting actions.

**Returns**:

  List of conflicting actions.

#### conflict\_has\_prior\_events

```python
@property
def conflict_has_prior_events() -> bool
```

Checks if prior events exist.

**Returns**:

  `True` if anything has happened before this conflict, otherwise `False`.

## TrackerEventStateTuple Objects

```python
class TrackerEventStateTuple(NamedTuple)
```

Holds a tracker, an event, and sliced states associated with those.

#### sliced\_states\_hash

```python
@property
def sliced_states_hash() -> int
```

Returns the hash of the sliced states.

#### find\_story\_conflicts

```python
def find_story_conflicts(trackers: List[TrackerWithCachedStates], domain: Domain, max_history: Optional[int] = None) -> List[StoryConflict]
```

Generates `StoryConflict` objects, describing conflicts in the given trackers.

**Arguments**:

- `trackers` - Trackers in which to search for conflicts.
- `domain` - The domain.
- `max_history` - The maximum history length to be taken into account.
  

**Returns**:

  StoryConflict objects.

