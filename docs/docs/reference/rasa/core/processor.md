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
async def handle_message(message: UserMessage) -> Optional[List[Dict[Text, Any]]]
```

Handle a single message with this processor.

#### predict\_next

```python
async def predict_next(sender_id: Text) -> Optional[Dict[Text, Any]]
```

Predict the next action for the current conversation state.

**Arguments**:

- `sender_id` - Conversation ID.
  

**Returns**:

  The prediction for the next action. `None` if no domain or policies loaded.

#### predict\_next\_with\_tracker

```python
def predict_next_with_tracker(tracker: DialogueStateTracker, verbosity: EventVerbosity = EventVerbosity.AFTER_RESTART) -> Optional[Dict[Text, Any]]
```

Predict the next action for a given conversation state.

**Arguments**:

- `tracker` - A tracker representing a conversation state.
- `verbosity` - Verbosity for the returned conversation state.
  

**Returns**:

  The prediction for the next action. `None` if no domain or policies loaded.

#### fetch\_tracker\_and\_update\_session

```python
async def fetch_tracker_and_update_session(sender_id: Text, output_channel: Optional[OutputChannel] = None, metadata: Optional[Dict] = None) -> DialogueStateTracker
```

Fetches tracker for `sender_id` and updates its conversation session.

If a new tracker is created, `action_session_start` is run.

**Arguments**:

- `metadata` - Data sent from client associated with the incoming user message.
- `output_channel` - Output channel associated with the incoming user message.
- `sender_id` - Conversation ID for which to fetch the tracker.
  

**Returns**:

  Tracker for `sender_id`.

#### fetch\_tracker\_with\_initial\_session

```python
async def fetch_tracker_with_initial_session(sender_id: Text, output_channel: Optional[OutputChannel] = None, metadata: Optional[Dict] = None) -> DialogueStateTracker
```

Fetches tracker for `sender_id` and runs a session start if it&#x27;s a new
tracker.

**Arguments**:

- `metadata` - Data sent from client associated with the incoming user message.
- `output_channel` - Output channel associated with the incoming user message.
- `sender_id` - Conversation ID for which to fetch the tracker.
  

**Returns**:

  Tracker for `sender_id`.

#### get\_tracker

```python
def get_tracker(conversation_id: Text) -> DialogueStateTracker
```

Get the tracker for a conversation.

In contrast to `fetch_tracker_and_update_session` this does not add any
`action_session_start` or `session_start` events at the beginning of a
conversation.

**Arguments**:

- `conversation_id` - The ID of the conversation for which the history should be
  retrieved.
  

**Returns**:

  Tracker for the conversation. Creates an empty tracker in case it&#x27;s a new
  conversation.

#### get\_trackers\_for\_all\_conversation\_sessions

```python
def get_trackers_for_all_conversation_sessions(conversation_id: Text) -> List[DialogueStateTracker]
```

Fetches all trackers for a conversation.

Individual trackers are returned for each conversation session found
for `conversation_id`.

**Arguments**:

- `conversation_id` - The ID of the conversation for which the trackers should
  be retrieved.
  

**Returns**:

  Trackers for the conversation.

#### log\_message

```python
async def log_message(message: UserMessage, should_save_tracker: bool = True) -> DialogueStateTracker
```

Log `message` on tracker belonging to the message&#x27;s conversation_id.

Optionally save the tracker if `should_save_tracker` is `True`. Tracker saving
can be skipped if the tracker returned by this method is used for further
processing and saved at a later stage.

#### execute\_action

```python
async def execute_action(sender_id: Text, action_name: Text, output_channel: OutputChannel, nlg: NaturalLanguageGenerator, prediction: PolicyPrediction) -> Optional[DialogueStateTracker]
```

Execute an action for a conversation.

Note that this might lead to unexpected bot behavior. Rather use an intent
to execute certain behavior within a conversation (e.g. by using
`trigger_external_user_uttered`).

**Arguments**:

- `sender_id` - The ID of the conversation.
- `action_name` - The name of the action which should be executed.
- `output_channel` - The output channel which should be used for bot responses.
- `nlg` - The response generator.
- `prediction` - The prediction for the action.
  

**Returns**:

  The new conversation state. Note that the new state is also persisted.

#### predict\_next\_action

```python
def predict_next_action(tracker: DialogueStateTracker) -> Tuple[rasa.core.actions.action.Action, PolicyPrediction]
```

Predicts the next action the bot should take after seeing x.

This should be overwritten by more advanced policies to use
ML to predict the action.

**Returns**:

  The index of the next action and prediction of the policy.
  

**Raises**:

  ActionLimitReached if the limit of actions to predict has been reached.

#### handle\_reminder

```python
async def handle_reminder(reminder_event: ReminderScheduled, sender_id: Text, output_channel: OutputChannel) -> None
```

Handle a reminder that is triggered asynchronously.

#### trigger\_external\_user\_uttered

```python
async def trigger_external_user_uttered(intent_name: Text, entities: Optional[Union[List[Dict[Text, Any]], Dict[Text, Text]]], tracker: DialogueStateTracker, output_channel: OutputChannel) -> None
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
async def parse_message(message: UserMessage, tracker: Optional[DialogueStateTracker] = None) -> Dict[Text, Any]
```

Interprete the passed message using the NLU interpreter.

**Arguments**:

- `message` - Message to handle
- `tracker` - Dialogue context of the message
  

**Returns**:

  Parsed data extracted from the message.

#### is\_action\_limit\_reached

```python
def is_action_limit_reached(tracker: DialogueStateTracker, should_predict_another_action: bool) -> bool
```

Check whether the maximum number of predictions has been met.

**Arguments**:

- `tracker` - instance of DialogueStateTracker.
- `should_predict_another_action` - Whether the last executed action allows
  for more actions to be predicted or not.
  

**Returns**:

  `True` if the limit of actions to predict has been reached.

#### should\_predict\_another\_action

```python
@staticmethod
def should_predict_another_action(action_name: Text) -> bool
```

Determine whether the processor should predict another action.

**Arguments**:

- `action_name` - Name of the latest executed action.
  

**Returns**:

  `False` if `action_name` is `ACTION_LISTEN_NAME` or
  `ACTION_SESSION_START_NAME`, otherwise `True`.

#### execute\_side\_effects

```python
async def execute_side_effects(events: List[Event], tracker: DialogueStateTracker, output_channel: OutputChannel) -> None
```

Send bot messages, schedule and cancel reminders that are logged
in the events array.

