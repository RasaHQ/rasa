---
sidebar_label: rasa.core.training.generator
title: rasa.core.training.generator
---
## TrackerWithCachedStates Objects

```python
class TrackerWithCachedStates(DialogueStateTracker)
```

A tracker wrapper that caches the state creation of the tracker.

#### clear\_states

```python
 | clear_states() -> None
```

Reset the states.

#### init\_copy

```python
 | init_copy() -> "TrackerWithCachedStates"
```

Create a new state tracker with the same initial values.

#### copy

```python
 | copy(sender_id: Text = "", sender_source: Text = "") -> "TrackerWithCachedStates"
```

Creates a duplicate of this tracker.

A new tracker will be created and all events
will be replayed.

#### update

```python
 | update(event: Event, skip_states: bool = False) -> None
```

Modify the state of the tracker according to an ``Event``.

## TrainingDataGenerator Objects

```python
class TrainingDataGenerator()
```

#### \_\_init\_\_

```python
 | __init__(story_graph: StoryGraph, domain: Domain, remove_duplicates: bool = True, unique_last_num_states: Optional[int] = None, augmentation_factor: int = 50, tracker_limit: Optional[int] = None, use_story_concatenation: bool = True, debug_plots: bool = False)
```

Given a set of story parts, generates all stories that are possible.

The different story parts can end and start with checkpoints
and this generator will match start and end checkpoints to
connect complete stories. Afterwards, duplicate stories will be
removed and the data is augmented (if augmentation is enabled).

#### generate

```python
 | generate() -> List[TrackerWithCachedStates]
```

Generate trackers from stories and rules.

**Returns**:

  The generated trackers.

#### generate\_story\_trackers

```python
 | generate_story_trackers() -> List[TrackerWithCachedStates]
```

Generate trackers from stories (exclude rule trackers).

**Returns**:

  The generated story trackers.

