---
sidebar_label: rasa.shared.core.events
title: rasa.shared.core.events
---
#### deserialise\_events

```python
def deserialise_events(serialized_events: List[Dict[Text, Any]]) -> List["Event"]
```

Convert a list of dictionaries to a list of corresponding events.

Example format:
    [{&quot;event&quot;: &quot;slot&quot;, &quot;value&quot;: 5, &quot;name&quot;: &quot;my_slot&quot;}]

#### format\_message

```python
def format_message(text: Text, intent: Optional[Text], entities: Union[Text, List[Any]]) -> Text
```

Uses NLU parser information to generate a message with inline entity annotations.

**Arguments**:

- `text` - text of the message
- `intent` - intent of the message
- `entities` - entities of the message
  

**Returns**:

  Message with entities annotated inline, e.g.
  `I am from [Berlin]{&quot;entity&quot;: &quot;city&quot;}`.

#### split\_events

```python
def split_events(events: Iterable["Event"], event_type_to_split_on: Type["Event"], additional_splitting_conditions: Optional[Dict[Text, Any]] = None, include_splitting_event: bool = True) -> List[List["Event"]]
```

Splits events according to an event type and condition.

**Examples**:

  Splitting events according to the event type `ActionExecuted` and the
  `action_name` &#x27;action_session_start&#x27; would look as follows:
  
  &gt;&gt; _events = split_events(
  events,
  ActionExecuted,
- `{&quot;action_name&quot;` - &quot;action_session_start&quot;},
  True
  )
  

**Arguments**:

- `events` - Events to split.
- `event_type_to_split_on` - The event type to split on.
- `additional_splitting_conditions` - Additional event attributes to split on.
- `include_splitting_event` - Whether the events of the type on which the split
  is based should be included in the returned events.
  

**Returns**:

  The split events.

#### do\_events\_begin\_with\_session\_start

```python
def do_events_begin_with_session_start(events: List["Event"]) -> bool
```

Determines whether `events` begins with a session start sequence.

A session start sequence is a sequence of two events: an executed
`action_session_start` as well as a logged `session_started`.

**Arguments**:

- `events` - The events to inspect.
  

**Returns**:

  Whether or not `events` begins with a session start sequence.

## Event Objects

```python
class Event(ABC)
```

Describes events in conversation and how the affect the conversation state.

Immutable representation of everything which happened during a conversation of the
user with the assistant. Tells the `rasa.shared.core.trackers.DialogueStateTracker`
how to update its state as the events occur.

#### as\_story\_string

```python
@abc.abstractmethod
def as_story_string() -> Optional[Text]
```

Returns the event as story string.

**Returns**:

  textual representation of the event or None.

#### fingerprint

```python
def fingerprint() -> Text
```

Returns a unique hash for the event which is stable across python runs.

**Returns**:

  fingerprint of the event

#### resolve\_by\_type

```python
@staticmethod
def resolve_by_type(type_name: Text, default: Optional[Type["Event"]] = None) -> Optional[Type["Event"]]
```

Returns a slots class by its type name.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

**Arguments**:

- `tracker` - The current conversation state.

#### \_\_eq\_\_

```python
@abc.abstractmethod
def __eq__(other: Any) -> bool
```

Compares object with other object.

#### \_\_str\_\_

```python
def __str__() -> Text
```

Returns text representation of event.

## AlwaysEqualEventMixin Objects

```python
class AlwaysEqualEventMixin(Event,  ABC)
```

Class to deduplicate common behavior for events without additional attributes.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Compares object with other object.

## SkipEventInMDStoryMixin Objects

```python
class SkipEventInMDStoryMixin(Event,  ABC)
```

Skips the visualization of an event in Markdown stories.

#### as\_story\_string

```python
def as_story_string() -> None
```

Returns the event as story string.

**Returns**:

  None, as this event should not appear inside the story.

## UserUttered Objects

```python
class UserUttered(Event)
```

The user has said something to the bot.

As a side effect a new `Turn` will be created in the `Tracker`.

#### \_\_init\_\_

```python
def __init__(text: Optional[Text] = None, intent: Optional[Dict] = None, entities: Optional[List[Dict]] = None, parse_data: Optional["NLUPredictionData"] = None, timestamp: Optional[float] = None, input_channel: Optional[Text] = None, message_id: Optional[Text] = None, metadata: Optional[Dict] = None, use_text_for_featurization: Optional[bool] = None) -> None
```

Creates event for incoming user message.

**Arguments**:

- `text` - Text of user message.
- `intent` - Intent prediction of user message.
- `entities` - Extracted entities.
- `parse_data` - Detailed NLU parsing result for message.
- `timestamp` - When the event was created.
- `metadata` - Additional event metadata.
- `input_channel` - Which channel the user used to send message.
- `message_id` - Unique ID for message.
- `use_text_for_featurization` - `True` if the message&#x27;s text was used to predict
  next action. `False` if the message&#x27;s intent was used.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash of object.

#### intent\_name

```python
@property
def intent_name() -> Optional[Text]
```

Returns intent name or `None` if no intent.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Compares object with other object.

#### \_\_str\_\_

```python
def __str__() -> Text
```

Returns text representation of event.

#### as\_sub\_state

```python
def as_sub_state() -> Dict[Text, Union[None, Text, List[Optional[Text]]]]
```

Turns a UserUttered event into features.

The substate contains information about entities, intent and text of the
`UserUttered` event.

**Returns**:

  a dictionary with intent name, text and entities

#### as\_story\_string

```python
def as_story_string(e2e: bool = False) -> Text
```

Return event as string for Markdown training format.

**Arguments**:

- `e2e` - `True` if the the event should be printed in the format for
  end-to-end conversation tests.
  

**Returns**:

  Event as string.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to tracker. See docstring of `Event`.

## DefinePrevUserUtteredFeaturization Objects

```python
class DefinePrevUserUtteredFeaturization(SkipEventInMDStoryMixin)
```

Stores information whether action was predicted based on text or intent.

#### \_\_init\_\_

```python
def __init__(use_text_for_featurization: bool, timestamp: Optional[float] = None, metadata: Optional[Dict[Text, Any]] = None) -> None
```

Creates event.

**Arguments**:

- `use_text_for_featurization` - `True` if message text was used to predict
  action. `False` if intent was used.
- `timestamp` - When the event was created.
- `metadata` - Additional event metadata.

#### \_\_str\_\_

```python
def __str__() -> Text
```

Returns text representation of event.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### as\_dict

```python
def as_dict() -> Dict[Text, Any]
```

Returns serialized event.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

**Arguments**:

- `tracker` - The current conversation state.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Compares object with other object.

## EntitiesAdded Objects

```python
class EntitiesAdded(SkipEventInMDStoryMixin)
```

Event that is used to add extracted entities to the tracker state.

#### \_\_init\_\_

```python
def __init__(entities: List[Dict[Text, Any]], timestamp: Optional[float] = None, metadata: Optional[Dict[Text, Any]] = None) -> None
```

Initializes event.

**Arguments**:

- `entities` - Entities extracted from previous user message. This can either
  be done by NLU components or end-to-end policy predictions.
- `timestamp` - the timestamp
- `metadata` - some optional metadata

#### \_\_str\_\_

```python
def __str__() -> Text
```

Returns the string representation of the event.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns the hash value of the event.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Compares this event with another event.

#### as\_dict

```python
def as_dict() -> Dict[Text, Any]
```

Converts the event into a dict.

**Returns**:

  A dict that represents this event.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

**Arguments**:

- `tracker` - The current conversation state.

## BotUttered Objects

```python
class BotUttered(SkipEventInMDStoryMixin)
```

The bot has said something to the user.

This class is not used in the story training as it is contained in the

``ActionExecuted`` class. An entry is made in the ``Tracker``.

#### \_\_init\_\_

```python
def __init__(text: Optional[Text] = None, data: Optional[Dict] = None, metadata: Optional[Dict[Text, Any]] = None, timestamp: Optional[float] = None) -> None
```

Creates event for a bot response.

**Arguments**:

- `text` - Plain text which bot responded with.
- `data` - Additional data for more complex utterances (e.g. buttons).
- `timestamp` - When the event was created.
- `metadata` - Additional event metadata.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Compares object with other object.

#### \_\_str\_\_

```python
def __str__() -> Text
```

Returns text representation of event.

#### \_\_repr\_\_

```python
def __repr__() -> Text
```

Returns text representation of event for debugging.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

#### message

```python
def message() -> Dict[Text, Any]
```

Return the complete message as a dictionary.

#### empty

```python
@staticmethod
def empty() -> "BotUttered"
```

Creates an empty bot utterance.

#### as\_dict

```python
def as_dict() -> Dict[Text, Any]
```

Returns serialized event.

## SlotSet Objects

```python
class SlotSet(Event)
```

The user has specified their preference for the value of a `slot`.

Every slot has a name and a value. This event can be used to set a
value for a slot on a conversation.

As a side effect the `Tracker`&#x27;s slots will be updated so
that `tracker.slots[key]=value`.

#### \_\_init\_\_

```python
def __init__(key: Text, value: Optional[Any] = None, timestamp: Optional[float] = None, metadata: Optional[Dict[Text, Any]] = None) -> None
```

Creates event to set slot.

**Arguments**:

- `key` - Name of the slot which is set.
- `value` - Value to which slot is set.
- `timestamp` - When the event was created.
- `metadata` - Additional event metadata.

#### \_\_str\_\_

```python
def __str__() -> Text
```

Returns text representation of event.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Compares object with other object.

#### as\_story\_string

```python
def as_story_string() -> Text
```

Returns text representation of event.

#### as\_dict

```python
def as_dict() -> Dict[Text, Any]
```

Returns serialized event.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

## Restarted Objects

```python
class Restarted(AlwaysEqualEventMixin)
```

Conversation should start over &amp; history wiped.

Instead of deleting all events, this event can be used to reset the
trackers state (e.g. ignoring any past user messages &amp; resetting all
the slots).

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### as\_story\_string

```python
def as_story_string() -> Text
```

Returns text representation of event.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Resets the tracker and triggers a followup `ActionSessionStart`.

## UserUtteranceReverted Objects

```python
class UserUtteranceReverted(AlwaysEqualEventMixin)
```

Bot reverts everything until before the most recent user message.

The bot will revert all events after the latest `UserUttered`, this
also means that the last event on the tracker is usually `action_listen`
and the bot is waiting for a new user message.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### as\_story\_string

```python
def as_story_string() -> Text
```

Returns text representation of event.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

## AllSlotsReset Objects

```python
class AllSlotsReset(AlwaysEqualEventMixin)
```

All Slots are reset to their initial values.

If you want to keep the dialogue history and only want to reset the
slots, you can use this event to set all the slots to their initial
values.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### as\_story\_string

```python
def as_story_string() -> Text
```

Returns text representation of event.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

## ReminderScheduled Objects

```python
class ReminderScheduled(Event)
```

Schedules the asynchronous triggering of a user intent at a given time.

The triggered intent can include entities if needed.

#### \_\_init\_\_

```python
def __init__(intent: Text, trigger_date_time: datetime, entities: Optional[List[Dict]] = None, name: Optional[Text] = None, kill_on_user_message: bool = True, timestamp: Optional[float] = None, metadata: Optional[Dict[Text, Any]] = None) -> None
```

Creates the reminder.

**Arguments**:

- `intent` - Name of the intent to be triggered.
- `trigger_date_time` - Date at which the execution of the action
  should be triggered (either utc or with tz).
- `name` - ID of the reminder. If there are multiple reminders with
  the same id only the last will be run.
- `entities` - Entities that should be supplied together with the
  triggered intent.
- `kill_on_user_message` - ``True`` means a user message before the
  trigger date will abort the reminder.
- `timestamp` - Creation date of the event.
- `metadata` - Optional event metadata.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Compares object with other object.

#### \_\_str\_\_

```python
def __str__() -> Text
```

Returns text representation of event.

#### as\_story\_string

```python
def as_story_string() -> Text
```

Returns text representation of event.

#### as\_dict

```python
def as_dict() -> Dict[Text, Any]
```

Returns serialized event.

## ReminderCancelled Objects

```python
class ReminderCancelled(Event)
```

Cancel certain jobs.

#### \_\_init\_\_

```python
def __init__(name: Optional[Text] = None, intent: Optional[Text] = None, entities: Optional[List[Dict]] = None, timestamp: Optional[float] = None, metadata: Optional[Dict[Text, Any]] = None) -> None
```

Creates a ReminderCancelled event.

If all arguments are `None`, this will cancel all reminders.
are to be cancelled. If no arguments are supplied, this will cancel all
reminders.

**Arguments**:

- `name` - Name of the reminder to be cancelled.
- `intent` - Intent name that is to be used to identify the reminders to be
  cancelled.
- `entities` - Entities that are to be used to identify the reminders to be
  cancelled.
- `timestamp` - Optional timestamp.
- `metadata` - Optional event metadata.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Compares object with other object.

#### \_\_str\_\_

```python
def __str__() -> Text
```

Returns text representation of event.

#### cancels\_job\_with\_name

```python
def cancels_job_with_name(job_name: Text, sender_id: Text) -> bool
```

Determines if this event should cancel the job with the given name.

**Arguments**:

- `job_name` - Name of the job to be tested.
- `sender_id` - The `sender_id` of the tracker.
  

**Returns**:

  `True`, if this `ReminderCancelled` event should cancel the job with the
  given name, and `False` otherwise.

#### as\_story\_string

```python
def as_story_string() -> Text
```

Returns text representation of event.

## ActionReverted Objects

```python
class ActionReverted(AlwaysEqualEventMixin)
```

Bot undoes its last action.

The bot reverts everything until before the most recent action.
This includes the action itself, as well as any events that
action created, like set slot events - the bot will now
predict a new action using the state before the most recent
action.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### as\_story\_string

```python
def as_story_string() -> Text
```

Returns text representation of event.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

## StoryExported Objects

```python
class StoryExported(Event)
```

Story should get dumped to a file.

#### \_\_init\_\_

```python
def __init__(path: Optional[Text] = None, timestamp: Optional[float] = None, metadata: Optional[Dict[Text, Any]] = None) -> None
```

Creates event about story exporting.

**Arguments**:

- `path` - Path to which story was exported to.
- `timestamp` - When the event was created.
- `metadata` - Additional event metadata.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### as\_story\_string

```python
def as_story_string() -> Text
```

Returns text representation of event.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Compares object with other object.

## FollowupAction Objects

```python
class FollowupAction(Event)
```

Enqueue a followup action.

#### \_\_init\_\_

```python
def __init__(name: Text, timestamp: Optional[float] = None, metadata: Optional[Dict[Text, Any]] = None) -> None
```

Creates an event which forces the model to run a certain action next.

**Arguments**:

- `name` - Name of the action to run.
- `timestamp` - When the event was created.
- `metadata` - Additional event metadata.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Compares object with other object.

#### \_\_str\_\_

```python
def __str__() -> Text
```

Returns text representation of event.

#### as\_story\_string

```python
def as_story_string() -> Text
```

Returns text representation of event.

#### as\_dict

```python
def as_dict() -> Dict[Text, Any]
```

Returns serialized event.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

## ConversationPaused Objects

```python
class ConversationPaused(AlwaysEqualEventMixin)
```

Ignore messages from the user to let a human take over.

As a side effect the `Tracker`&#x27;s `paused` attribute will
be set to `True`.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### as\_story\_string

```python
def as_story_string() -> Text
```

Returns text representation of event.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

## ConversationResumed Objects

```python
class ConversationResumed(AlwaysEqualEventMixin)
```

Bot takes over conversation.

Inverse of `PauseConversation`. As a side effect the `Tracker`&#x27;s
`paused` attribute will be set to `False`.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### as\_story\_string

```python
def as_story_string() -> Text
```

Returns text representation of event.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

## ActionExecuted Objects

```python
class ActionExecuted(Event)
```

An operation describes an action taken + its result.

It comprises an action and a list of events. operations will be appended
to the latest `Turn`` in `Tracker.turns`.

#### \_\_init\_\_

```python
def __init__(action_name: Optional[Text] = None, policy: Optional[Text] = None, confidence: Optional[float] = None, timestamp: Optional[float] = None, metadata: Optional[Dict] = None, action_text: Optional[Text] = None, hide_rule_turn: bool = False) -> None
```

Creates event for a successful event execution.

**Arguments**:

- `action_name` - Name of the action which was executed. `None` if it was an
  end-to-end prediction.
- `policy` - Policy which predicted action.
- `confidence` - Confidence with which policy predicted action.
- `timestamp` - When the event was created.
- `metadata` - Additional event metadata.
- `action_text` - In case it&#x27;s an end-to-end action prediction, the text which
  was predicted.
- `hide_rule_turn` - If `True`, this action should be hidden in the dialogue
  history created for ML-based policies.

#### \_\_repr\_\_

```python
def __repr__() -> Text
```

Returns event as string for debugging.

#### \_\_str\_\_

```python
def __str__() -> Optional[Text]
```

Returns event as human readable string.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Compares object with other object.

#### as\_story\_string

```python
def as_story_string() -> Optional[Text]
```

Returns event in Markdown format.

#### as\_dict

```python
def as_dict() -> Dict[Text, Any]
```

Returns serialized event.

#### as\_sub\_state

```python
def as_sub_state() -> Dict[Text, Text]
```

Turns ActionExecuted into a dictionary containing action name or action text.

One action cannot have both set at the same time

**Returns**:

  a dictionary containing action name or action text with the corresponding
  key.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

## AgentUttered Objects

```python
class AgentUttered(SkipEventInMDStoryMixin)
```

The agent has said something to the user.

This class is not used in the story training as it is contained in the
``ActionExecuted`` class. An entry is made in the ``Tracker``.

#### \_\_init\_\_

```python
def __init__(text: Optional[Text] = None, data: Optional[Any] = None, timestamp: Optional[float] = None, metadata: Optional[Dict[Text, Any]] = None) -> None
```

See docstring of `BotUttered`.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Compares object with other object.

#### \_\_str\_\_

```python
def __str__() -> Text
```

Returns text representation of event.

#### as\_dict

```python
def as_dict() -> Dict[Text, Any]
```

Returns serialized event.

## ActiveLoop Objects

```python
class ActiveLoop(Event)
```

If `name` is given: activates a loop with `name` else deactivates active loop.

#### \_\_init\_\_

```python
def __init__(name: Optional[Text], timestamp: Optional[float] = None, metadata: Optional[Dict[Text, Any]] = None) -> None
```

Creates event for active loop.

**Arguments**:

- `name` - Name of activated loop or `None` if current loop is deactivated.
- `timestamp` - When the event was created.
- `metadata` - Additional event metadata.

#### \_\_str\_\_

```python
def __str__() -> Text
```

Returns text representation of event.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Compares object with other object.

#### as\_story\_string

```python
def as_story_string() -> Text
```

Returns text representation of event.

#### as\_dict

```python
def as_dict() -> Dict[Text, Any]
```

Returns serialized event.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

## LegacyForm Objects

```python
class LegacyForm(ActiveLoop)
```

Legacy handler of old `Form` events.

The `ActiveLoop` event used to be called `Form`. This class is there to handle old
legacy events which were stored with the old type name `form`.

#### as\_dict

```python
def as_dict() -> Dict[Text, Any]
```

Returns serialized event.

#### fingerprint

```python
def fingerprint() -> Text
```

Returns the hash of the event.

## LoopInterrupted Objects

```python
class LoopInterrupted(SkipEventInMDStoryMixin)
```

Event added by FormPolicy and RulePolicy.

Notifies form action whether or not to validate the user input.

#### \_\_init\_\_

```python
def __init__(is_interrupted: bool, timestamp: Optional[float] = None, metadata: Optional[Dict[Text, Any]] = None) -> None
```

Event to notify that loop was interrupted.

This e.g. happens when a user is within a form, and is de-railing the
form-filling by asking FAQs.

**Arguments**:

- `is_interrupted` - `True` if the loop execution was interrupted, and ML
  policies had to take over the last prediction.
- `timestamp` - When the event was created.
- `metadata` - Additional event metadata.

#### \_\_str\_\_

```python
def __str__() -> Text
```

Returns text representation of event.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Compares object with other object.

#### as\_dict

```python
def as_dict() -> Dict[Text, Any]
```

Returns serialized event.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

## LegacyFormValidation Objects

```python
class LegacyFormValidation(LoopInterrupted)
```

Legacy handler of old `FormValidation` events.

The `LoopInterrupted` event used to be called `FormValidation`. This class is there
to handle old legacy events which were stored with the old type name
`form_validation`.

#### \_\_init\_\_

```python
def __init__(validate: bool, timestamp: Optional[float] = None, metadata: Optional[Dict[Text, Any]] = None) -> None
```

See parent class docstring.

#### as\_dict

```python
def as_dict() -> Dict[Text, Any]
```

Returns serialized event.

#### fingerprint

```python
def fingerprint() -> Text
```

Returns hash of the event.

## ActionExecutionRejected Objects

```python
class ActionExecutionRejected(SkipEventInMDStoryMixin)
```

Notify Core that the execution of the action has been rejected.

#### \_\_init\_\_

```python
def __init__(action_name: Text, policy: Optional[Text] = None, confidence: Optional[float] = None, timestamp: Optional[float] = None, metadata: Optional[Dict[Text, Any]] = None) -> None
```

Creates event.

**Arguments**:

- `action_name` - Action which was rejected.
- `policy` - Policy which predicted the rejected action.
- `confidence` - Confidence with which the reject action was predicted.
- `timestamp` - When the event was created.
- `metadata` - Additional event metadata.

#### \_\_str\_\_

```python
def __str__() -> Text
```

Returns text representation of event.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Compares object with other object.

#### as\_dict

```python
def as_dict() -> Dict[Text, Any]
```

Returns serialized event.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

## SessionStarted Objects

```python
class SessionStarted(AlwaysEqualEventMixin)
```

Mark the beginning of a new conversation session.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Returns unique hash for event.

#### as\_story\_string

```python
def as_story_string() -> None
```

Skips representing event in stories.

#### apply\_to

```python
def apply_to(tracker: "DialogueStateTracker") -> None
```

Applies event to current conversation state.

