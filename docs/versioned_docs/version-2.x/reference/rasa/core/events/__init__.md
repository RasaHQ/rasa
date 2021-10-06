---
sidebar_label: rasa.core.events
title: rasa.core.events
---
#### deserialise\_events

```python
deserialise_events(serialized_events: List[Dict[Text, Any]]) -> List["Event"]
```

Convert a list of dictionaries to a list of corresponding events.

Example format:
    [{&quot;event&quot;: &quot;slot&quot;, &quot;value&quot;: 5, &quot;name&quot;: &quot;my_slot&quot;}]

#### md\_format\_message

```python
md_format_message(text: Text, intent: Optional[Text], entities: Union[Text, List[Any]]) -> Text
```

Uses NLU parser information to generate a message with inline entity annotations.

**Arguments**:

- `text` - text of the message
- `intent` - intent of the message
- `entities` - entities of the message
  

**Returns**:

  Message with entities annotated inline, e.g.
  `I am from [Berlin]{&quot;entity&quot;: &quot;city&quot;}`.

## Event Objects

```python
class Event()
```

Events describe everything that occurs in
a conversation and tell the :class:`rasa.core.trackers.DialogueStateTracker`
how to update its state.

#### resolve\_by\_type

```python
 | @staticmethod
 | resolve_by_type(type_name: Text, default: Optional[Type["Event"]] = None) -> Optional[Type["Event"]]
```

Returns a slots class by its type name.

## UserUttered Objects

```python
class UserUttered(Event)
```

The user has said something to the bot.

As a side effect a new ``Turn`` will be created in the ``Tracker``.

#### as\_sub\_state

```python
 | as_sub_state() -> Dict[Text, Union[None, Text, List[Optional[Text]]]]
```

Turns a UserUttered event into a substate containing information about entities,
intent and text of the UserUttered

**Returns**:

  a dictionary with intent name, text and entities

## BotUttered Objects

```python
class BotUttered(Event)
```

The bot has said something to the user.

This class is not used in the story training as it is contained in the

``ActionExecuted`` class. An entry is made in the ``Tracker``.

#### message

```python
 | message() -> Dict[Text, Any]
```

Return the complete message as a dictionary.

## SlotSet Objects

```python
class SlotSet(Event)
```

The user has specified their preference for the value of a ``slot``.

Every slot has a name and a value. This event can be used to set a
value for a slot on a conversation.

As a side effect the ``Tracker``&#x27;s slots will be updated so
that ``tracker.slots[key]=value``.

## Restarted Objects

```python
class Restarted(Event)
```

Conversation should start over &amp; history wiped.

Instead of deleting all events, this event can be used to reset the
trackers state (e.g. ignoring any past user messages &amp; resetting all
the slots).

## UserUtteranceReverted Objects

```python
class UserUtteranceReverted(Event)
```

Bot reverts everything until before the most recent user message.

The bot will revert all events after the latest `UserUttered`, this
also means that the last event on the tracker is usually `action_listen`
and the bot is waiting for a new user message.

## AllSlotsReset Objects

```python
class AllSlotsReset(Event)
```

All Slots are reset to their initial values.

If you want to keep the dialogue history and only want to reset the
slots, you can use this event to set all the slots to their initial
values.

## ReminderScheduled Objects

```python
class ReminderScheduled(Event)
```

Schedules the asynchronous triggering of a user intent
(with entities if needed) at a given time.

#### \_\_init\_\_

```python
 | __init__(intent: Text, trigger_date_time: datetime, entities: Optional[List[Dict]] = None, name: Optional[Text] = None, kill_on_user_message: bool = True, timestamp: Optional[float] = None, metadata: Optional[Dict[Text, Any]] = None) -> None
```

Creates the reminder

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

## ReminderCancelled Objects

```python
class ReminderCancelled(Event)
```

Cancel certain jobs.

#### \_\_init\_\_

```python
 | __init__(name: Optional[Text] = None, intent: Optional[Text] = None, entities: Optional[List[Dict]] = None, timestamp: Optional[float] = None, metadata: Optional[Dict[Text, Any]] = None) -> None
```

Creates a ReminderCancelled event.

If all arguments are `None`, this will cancel all reminders.
are to be cancelled. If no arguments are supplied, this will cancel all reminders.

**Arguments**:

- `name` - Name of the reminder to be cancelled.
- `intent` - Intent name that is to be used to identify the reminders to be cancelled.
- `entities` - Entities that are to be used to identify the reminders to be cancelled.
- `timestamp` - Optional timestamp.
- `metadata` - Optional event metadata.

#### cancels\_job\_with\_name

```python
 | cancels_job_with_name(job_name: Text, sender_id: Text) -> bool
```

Determines if this `ReminderCancelled` event should cancel the job with the given name.

**Arguments**:

- `job_name` - Name of the job to be tested.
- `sender_id` - The `sender_id` of the tracker.
  

**Returns**:

  `True`, if this `ReminderCancelled` event should cancel the job with the given name,
  and `False` otherwise.

## ActionReverted Objects

```python
class ActionReverted(Event)
```

Bot undoes its last action.

The bot reverts everything until before the most recent action.
This includes the action itself, as well as any events that
action created, like set slot events - the bot will now
predict a new action using the state before the most recent
action.

## StoryExported Objects

```python
class StoryExported(Event)
```

Story should get dumped to a file.

## FollowupAction Objects

```python
class FollowupAction(Event)
```

Enqueue a followup action.

## ConversationPaused Objects

```python
class ConversationPaused(Event)
```

Ignore messages from the user to let a human take over.

As a side effect the ``Tracker``&#x27;s ``paused`` attribute will
be set to ``True``.

## ConversationResumed Objects

```python
class ConversationResumed(Event)
```

Bot takes over conversation.

Inverse of ``PauseConversation``. As a side effect the ``Tracker``&#x27;s
``paused`` attribute will be set to ``False``.

## ActionExecuted Objects

```python
class ActionExecuted(Event)
```

An operation describes an action taken + its result.

It comprises an action and a list of events. operations will be appended
to the latest `Turn`` in `Tracker.turns`.

#### as\_sub\_state

```python
 | as_sub_state() -> Dict[Text, Text]
```

Turns ActionExecuted into a dictionary containing action name or action text.
One action cannot have both set at the same time

**Returns**:

  a dictionary containing action name or action text with the corresponding key

## AgentUttered Objects

```python
class AgentUttered(Event)
```

The agent has said something to the user.

This class is not used in the story training as it is contained in the
``ActionExecuted`` class. An entry is made in the ``Tracker``.

## ActiveLoop Objects

```python
class ActiveLoop(Event)
```

If `name` is not None: activates a loop with `name` else deactivates active loop.

## LegacyForm Objects

```python
class LegacyForm(ActiveLoop)
```

Legacy handler of old `Form` events.

The `ActiveLoop` event used to be called `Form`. This class is there to handle old
legacy events which were stored with the old type name `form`.

## FormValidation Objects

```python
class FormValidation(Event)
```

Event added by FormPolicy and RulePolicy to notify form action
whether or not to validate the user input.

## ActionExecutionRejected Objects

```python
class ActionExecutionRejected(Event)
```

Notify Core that the execution of the action has been rejected

## SessionStarted Objects

```python
class SessionStarted(Event)
```

Mark the beginning of a new conversation session.

