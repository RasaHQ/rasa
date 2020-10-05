---
sidebar_label: rasa.core.processor
title: rasa.core.processor
---

## MessageProcessor Objects

```python
class MessageProcessor()
```

#### handle\_message

```python
 | async handle_message(message: UserMessage) -> Optional[List[Dict[Text, Any]]]
```

Handle a single message with this processor.

#### get\_tracker\_with\_session\_start

```python
 | async get_tracker_with_session_start(sender_id: Text, output_channel: Optional[OutputChannel] = None, metadata: Optional[Dict] = None) -> Optional[DialogueStateTracker]
```

Get tracker for `sender_id` or create a new tracker for `sender_id`.

If a new tracker is created, `action_session_start` is run.

**Arguments**:

- `metadata` - Data sent from client associated with the incoming user message.
- `output_channel` - Output channel associated with the incoming user message.
- `sender_id` - Conversation ID for which to fetch the tracker.
  

**Returns**:

  Tracker for `sender_id` if available, `None` otherwise.

#### get\_tracker

```python
 | get_tracker(conversation_id: Text) -> Optional[DialogueStateTracker]
```

Get the tracker for a conversation.

In contrast to `get_tracker_with_session_start` this does not add any
`action_session_start` or `session_start` events at the beginning of a
conversation.

**Arguments**:

- `conversation_id` - The ID of the conversation for which the history should be
  retrieved.
  

**Returns**:

  Tracker for the conversation. Creates an empty tracker in case it&#x27;s a new
  conversation.

#### log\_message

```python
 | async log_message(message: UserMessage, should_save_tracker: bool = True) -> Optional[DialogueStateTracker]
```

Log `message` on tracker belonging to the message&#x27;s conversation_id.

Optionally save the tracker if `should_save_tracker` is `True`. Tracker saving
can be skipped if the tracker returned by this method is used for further
processing and saved at a later stage.

#### predict\_next\_action

```python
 | predict_next_action(tracker: DialogueStateTracker) -> Tuple[rasa.core.actions.action.Action, Optional[Text], float]
```

Predicts the next action the bot should take after seeing x.

This should be overwritten by more advanced policies to use
ML to predict the action. Returns the index of the next action.

#### handle\_reminder

```python
 | async handle_reminder(reminder_event: ReminderScheduled, sender_id: Text, output_channel: OutputChannel) -> None
```

Handle a reminder that is triggered asynchronously.

#### trigger\_external\_user\_uttered

```python
 | async trigger_external_user_uttered(intent_name: Text, entities: Optional[Union[List[Dict[Text, Any]], Dict[Text, Text]]], tracker: DialogueStateTracker, output_channel: OutputChannel) -> None
```

Triggers an external message.

Triggers an external message (like a user message, but invisible;
used, e.g., by a reminder or the trigger_intent endpoint).

**Arguments**:

- `intent_name` - Name of the intent to be triggered.
- `entities` - Entities to be passed on.
- `tracker` - The tracker to which the event should be added.
- `output_channel` - The output channel.

#### parse\_message

```python
 | async parse_message(message: UserMessage, tracker: Optional[DialogueStateTracker] = None) -> Dict[Text, Any]
```

Interprete the passed message using the NLU interpreter.

**Arguments**:

- `message` - Message to handle
- `tracker` - Dialogue context of the message
  

**Returns**:

  Parsed data extracted from the message.

#### is\_action\_limit\_reached

```python
 | is_action_limit_reached(num_predicted_actions: int, should_predict_another_action: bool) -> bool
```

Check whether the maximum number of predictions has been met.

**Arguments**:

- `num_predicted_actions` - Number of predicted actions.
- `should_predict_another_action` - Whether the last executed action allows
  for more actions to be predicted or not.
  

**Returns**:

  `True` if the limit of actions to predict has been reached.

#### should\_predict\_another\_action

```python
 | @staticmethod
 | should_predict_another_action(action_name: Text) -> bool
```

Determine whether the processor should predict another action.

**Arguments**:

- `action_name` - Name of the latest executed action.
  

**Returns**:

  `False` if `action_name` is `ACTION_LISTEN_NAME` or
  `ACTION_SESSION_START_NAME`, otherwise `True`.

#### execute\_side\_effects

```python
 | async execute_side_effects(events: List[Event], tracker: DialogueStateTracker, output_channel: OutputChannel) -> None
```

Send bot messages, schedule and cancel reminders that are logged
in the events array.

