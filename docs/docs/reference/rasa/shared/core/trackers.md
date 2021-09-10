---
sidebar_label: rasa.shared.core.trackers
title: rasa.shared.core.trackers
---
## EventVerbosity Objects

```python
class EventVerbosity(Enum)
```

Filter on which events to include in tracker dumps.

## AnySlotDict Objects

```python
class AnySlotDict(dict)
```

A slot dictionary that pretends every slot exists, by creating slots on demand.

This only uses the generic slot type! This means certain functionality wont work,
e.g. properly featurizing the slot.

## DialogueStateTracker Objects

```python
class DialogueStateTracker()
```

Maintains the state of a conversation.

The field max_event_history will only give you these last events,
it can be set in the tracker_store

#### from\_dict

```python
@classmethod
def from_dict(cls, sender_id: Text, events_as_dict: List[Dict[Text, Any]], slots: Optional[List[Slot]] = None, max_event_history: Optional[int] = None) -> "DialogueStateTracker"
```

Create a tracker from dump.

The dump should be an array of dumped events. When restoring
the tracker, these events will be replayed to recreate the state.

#### from\_events

```python
@classmethod
def from_events(cls, sender_id: Text, evts: List[Event], slots: Optional[Iterable[Slot]] = None, max_event_history: Optional[int] = None, sender_source: Optional[Text] = None, domain: Optional[Domain] = None) -> "DialogueStateTracker"
```

Creates tracker from existing events.

**Arguments**:

- `sender_id` - The ID of the conversation.
- `evts` - Existing events which should be applied to the new tracker.
- `slots` - Slots which can be set.
- `max_event_history` - Maximum number of events which should be stored.
- `sender_source` - File source of the messages.
- `domain` - The current model domain.
  

**Returns**:

  Instantiated tracker with its state updated according to the given
  events.

#### \_\_init\_\_

```python
def __init__(sender_id: Text, slots: Optional[Iterable[Slot]], max_event_history: Optional[int] = None, sender_source: Optional[Text] = None, is_rule_tracker: bool = False) -> None
```

Initialize the tracker.

A set of events can be stored externally, and we will run through all
of them to get the current state. The tracker will represent all the
information we captured while processing messages of the dialogue.

#### current\_state

```python
def current_state(event_verbosity: EventVerbosity = EventVerbosity.NONE) -> Dict[Text, Any]
```

Returns the current tracker state as an object.

#### freeze\_current\_state

```python
@staticmethod
def freeze_current_state(state: State) -> FrozenState
```

Convert State dict into a hashable format FrozenState.

**Arguments**:

- `state` - The state which should be converted
  

**Returns**:

  hashable form of the state of type `FrozenState`

#### past\_states

```python
def past_states(domain: Domain, omit_unset_slots: bool = False, ignore_rule_only_turns: bool = False, rule_only_data: Optional[Dict[Text, Any]] = None) -> List[State]
```

Generates the past states of this tracker based on the history.

**Arguments**:

- `domain` - The Domain.
- `omit_unset_slots` - If `True` do not include the initial values of slots.
- `ignore_rule_only_turns` - If True ignore dialogue turns that are present
  only in rules.
- `rule_only_data` - Slots and loops,
  which only occur in rules but not in stories.
  

**Returns**:

  A list of states

#### change\_loop\_to

```python
def change_loop_to(loop_name: Optional[Text]) -> None
```

Set the currently active loop.

**Arguments**:

- `loop_name` - The name of loop which should be marked as active.

#### interrupt\_loop

```python
def interrupt_loop(is_interrupted: bool) -> None
```

Interrupt loop and mark that we entered an unhappy path in the conversation.

**Arguments**:

- `is_interrupted` - `True` if the loop was run after an unhappy path.

#### reject\_action

```python
def reject_action(action_name: Text) -> None
```

Notify active loop that it was rejected.

#### set\_latest\_action

```python
def set_latest_action(action: Dict[Text, Text]) -> None
```

Sets latest action name or text.

Resets loop validation and rejection parameters.

**Arguments**:

- `action` - Serialized action event.

#### current\_slot\_values

```python
def current_slot_values() -> Dict[Text, Any]
```

Return the currently set values of the slots

#### get\_slot

```python
def get_slot(key: Text) -> Optional[Any]
```

Retrieves the value of a slot.

#### get\_latest\_entity\_values

```python
def get_latest_entity_values(entity_type: Text, entity_role: Optional[Text] = None, entity_group: Optional[Text] = None) -> Iterator[Text]
```

Get entity values found for the passed entity type and optional role and
group in latest message.

If you are only interested in the first entity of a given type use
`next(tracker.get_latest_entity_values(&quot;my_entity_name&quot;), None)`.
If no entity is found `None` is the default result.

**Arguments**:

- `entity_type` - the entity type of interest
- `entity_role` - optional entity role of interest
- `entity_group` - optional entity group of interest
  

**Returns**:

  Entity values.

#### get\_latest\_input\_channel

```python
def get_latest_input_channel() -> Optional[Text]
```

Get the name of the input_channel of the latest UserUttered event

#### is\_paused

```python
def is_paused() -> bool
```

State whether the tracker is currently paused.

#### idx\_after\_latest\_restart

```python
def idx_after_latest_restart() -> int
```

Return the idx of the most recent restart in the list of events.

If the conversation has not been restarted, ``0`` is returned.

#### events\_after\_latest\_restart

```python
def events_after_latest_restart() -> List[Event]
```

Return a list of events after the most recent restart.

#### init\_copy

```python
def init_copy() -> "DialogueStateTracker"
```

Creates a new state tracker with the same initial values.

#### generate\_all\_prior\_trackers

```python
def generate_all_prior_trackers() -> Generator[Tuple["DialogueStateTracker", bool], None, None]
```

Returns a generator of the previous trackers of this tracker.

**Returns**:

  The tuple with the tracker before each action,
  and the boolean flag representing whether this action should be hidden
  in the dialogue history created for ML-based policies.

#### applied\_events

```python
def applied_events() -> List[Event]
```

Returns all actions that should be applied - w/o reverted events.

**Returns**:

  The events applied to the tracker.

#### replay\_events

```python
def replay_events() -> None
```

Update the tracker based on a list of events.

#### recreate\_from\_dialogue

```python
def recreate_from_dialogue(dialogue: Dialogue) -> None
```

Use a serialised `Dialogue` to update the trackers state.

This uses the state as is persisted in a ``TrackerStore``. If the
tracker is blank before calling this method, the final state will be
identical to the tracker from which the dialogue was created.

#### copy

```python
def copy() -> "DialogueStateTracker"
```

Creates a duplicate of this tracker

#### travel\_back\_in\_time

```python
def travel_back_in_time(target_time: float) -> "DialogueStateTracker"
```

Creates a new tracker with a state at a specific timestamp.

A new tracker will be created and all events previous to the
passed time stamp will be replayed. Events that occur exactly
at the target time will be included.

#### as\_dialogue

```python
def as_dialogue() -> Dialogue
```

Return a ``Dialogue`` object containing all of the turns.

This can be serialised and later used to recover the state
of this tracker exactly.

#### update

```python
def update(event: Event, domain: Optional[Domain] = None) -> None
```

Modify the state of the tracker according to an ``Event``.

#### update\_with\_events

```python
def update_with_events(new_events: List[Event], domain: Optional[Domain], override_timestamp: bool = True) -> None
```

Adds multiple events to the tracker.

**Arguments**:

- `new_events` - Events to apply.
- `domain` - The current model&#x27;s domain.
- `override_timestamp` - If `True` refresh all timestamps of the events. As the
  events are usually created at some earlier point, this makes sure that
  all new events come after any current tracker events.

#### as\_story

```python
def as_story(include_source: bool = False) -> "Story"
```

Dump the tracker as a story in the Rasa Core story format.

Returns the dumped tracker as a string.

#### export\_stories

```python
def export_stories(writer: "StoryWriter", e2e: bool = False, include_source: bool = False, should_append_stories: bool = False) -> Text
```

Dump the tracker as a story in the Rasa Core story format.

**Returns**:

  The dumped tracker as a string.

#### export\_stories\_to\_file

```python
def export_stories_to_file(export_path: Text = "debug_stories.yml") -> None
```

Dump the tracker as a story to a file.

#### get\_last\_event\_for

```python
def get_last_event_for(event_type: Union[Type[Event], Tuple[Type, ...]], action_names_to_exclude: List[Text] = None, skip: int = 0, event_verbosity: EventVerbosity = EventVerbosity.APPLIED) -> Optional[Event]
```

Gets the last event of a given type which was actually applied.

**Arguments**:

- `event_type` - The type of event you want to find.
- `action_names_to_exclude` - Events of type `ActionExecuted` which
  should be excluded from the results. Can be used to skip
  `action_listen` events.
- `skip` - Skips n possible results before return an event.
- `event_verbosity` - Which `EventVerbosity` should be used to search for events.
  

**Returns**:

  event which matched the query or `None` if no event matched.

#### last\_executed\_action\_has

```python
def last_executed_action_has(name: Text, skip: int = 0) -> bool
```

Returns whether last `ActionExecuted` event had a specific name.

**Arguments**:

- `name` - Name of the event which should be matched.
- `skip` - Skips n possible results in between.
  

**Returns**:

  `True` if last executed action had name `name`, otherwise `False`.

#### trigger\_followup\_action

```python
def trigger_followup_action(action: Text) -> None
```

Triggers another action following the execution of the current.

#### clear\_followup\_action

```python
def clear_followup_action() -> None
```

Clears follow up action when it was executed.

#### active\_loop\_name

```python
@property
def active_loop_name() -> Optional[Text]
```

Get the name of the currently active loop.

Returns: `None` if no active loop or the name of the currently active loop.

#### latest\_action\_name

```python
@property
def latest_action_name() -> Optional[Text]
```

Get the name of the previously executed action or text of e2e action.

Returns: name of the previously executed action or text of e2e action

#### fingerprint

```python
def fingerprint() -> Text
```

Returns a unique hash for the tracker which is stable across python runs.

**Returns**:

  fingerprint of the tracker

#### get\_active\_loop\_name

```python
def get_active_loop_name(state: State) -> Optional[Union[Text, Tuple[Union[float, Text]]]]
```

Get the name of current active loop.

**Arguments**:

- `state` - The state from which the name of active loop should be extracted
  

**Returns**:

  the name of active loop or None

#### is\_prev\_action\_listen\_in\_state

```python
def is_prev_action_listen_in_state(state: State) -> bool
```

Check if action_listen is the previous executed action.

**Arguments**:

- `state` - The state for which the check should be performed
  

**Returns**:

  boolean value indicating whether action_listen is previous action

#### get\_trackers\_for\_conversation\_sessions

```python
def get_trackers_for_conversation_sessions(tracker: DialogueStateTracker) -> List[DialogueStateTracker]
```

Generate trackers for `tracker` that are split by conversation sessions.

**Arguments**:

- `tracker` - Instance of `DialogueStateTracker` to split.
  

**Returns**:

  The trackers split by conversation sessions.

