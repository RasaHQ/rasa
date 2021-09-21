---
sidebar_label: rasa.core.actions.action
title: rasa.core.actions.action
---
#### default\_actions

```python
def default_actions(action_endpoint: Optional[EndpointConfig] = None) -> List["Action"]
```

List default actions.

#### action\_for\_index

```python
def action_for_index(index: int, domain: Domain, action_endpoint: Optional[EndpointConfig]) -> "Action"
```

Get an action based on its index in the list of available actions.

**Arguments**:

- `index` - The index of the action. This is usually used by `Policy`s as they
  predict the action index instead of the name.
- `domain` - The `Domain` of the current model. The domain contains the actions
  provided by the user + the default actions.
- `action_endpoint` - Can be used to run `custom_actions`
  (e.g. using the `rasa-sdk`).
  

**Returns**:

  The instantiated `Action` or `None` if no `Action` was found for the given
  index.

#### is\_retrieval\_action

```python
def is_retrieval_action(action_name: Text, retrieval_intents: List[Text]) -> bool
```

Check if an action name is a retrieval action.

The name for a retrieval action has an extra `utter_` prefix added to
the corresponding retrieval intent name.

**Arguments**:

- `action_name` - Name of the action.
- `retrieval_intents` - List of retrieval intents defined in the NLU training data.
  

**Returns**:

  `True` if the resolved intent name is present in the list of retrieval
  intents, `False` otherwise.

#### action\_for\_name\_or\_text

```python
def action_for_name_or_text(action_name_or_text: Text, domain: Domain, action_endpoint: Optional[EndpointConfig]) -> "Action"
```

Retrieves an action by its name or by its text in case it&#x27;s an end-to-end action.

**Arguments**:

- `action_name_or_text` - The name of the action.
- `domain` - The current model domain.
- `action_endpoint` - The endpoint to execute custom actions.
  

**Raises**:

- `ActionNotFoundException` - If action not in current domain.
  

**Returns**:

  The instantiated action.

#### create\_bot\_utterance

```python
def create_bot_utterance(message: Dict[Text, Any]) -> BotUttered
```

Create BotUttered event from message.

## Action Objects

```python
class Action()
```

Next action to be taken in response to a dialogue state.

#### name

```python
def name() -> Text
```

Unique identifier of this simple action.

#### run

```python
async def run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Execute the side effects of this action.

**Arguments**:

- `nlg` - which ``nlg`` to use for response generation
- `output_channel` - ``output_channel`` to which to send the resulting message.
- `tracker` _DialogueStateTracker_ - the state tracker for the current
  user. You can access slot values using
  ``tracker.get_slot(slot_name)`` and the most recent user
  message is ``tracker.latest_message.text``.
- `domain` _Domain_ - the bot&#x27;s domain
  

**Returns**:

  A list of :class:`rasa.core.events.Event` instances

#### \_\_str\_\_

```python
def __str__() -> Text
```

Returns text representation of form.

#### event\_for\_successful\_execution

```python
def event_for_successful_execution(prediction: PolicyPrediction) -> ActionExecuted
```

Event which should be logged for the successful execution of this action.

**Arguments**:

- `prediction` - Prediction which led to the execution of this event.
  

**Returns**:

  Event which should be logged onto the tracker.

## ActionBotResponse Objects

```python
class ActionBotResponse(Action)
```

An action which only effect is to utter a response when it is run.

#### \_\_init\_\_

```python
def __init__(name: Text, silent_fail: Optional[bool] = False) -> None
```

Creates action.

**Arguments**:

- `name` - Name of the action.
- `silent_fail` - `True` if the action should fail silently in case no response
  was defined for this action.

#### run

```python
async def run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Simple run implementation uttering a (hopefully defined) response.

#### name

```python
def name() -> Text
```

Returns action name.

## ActionEndToEndResponse Objects

```python
class ActionEndToEndResponse(Action)
```

Action to utter end-to-end responses to the user.

#### \_\_init\_\_

```python
def __init__(action_text: Text) -> None
```

Creates action.

**Arguments**:

- `action_text` - Text of end-to-end bot response.

#### name

```python
def name() -> Text
```

Returns action name.

#### run

```python
async def run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Runs action (see parent class for full docstring).

#### event\_for\_successful\_execution

```python
def event_for_successful_execution(prediction: PolicyPrediction) -> ActionExecuted
```

Event which should be logged for the successful execution of this action.

**Arguments**:

- `prediction` - Prediction which led to the execution of this event.
  

**Returns**:

  Event which should be logged onto the tracker.

## ActionRetrieveResponse Objects

```python
class ActionRetrieveResponse(ActionBotResponse)
```

An action which queries the Response Selector for the appropriate response.

#### \_\_init\_\_

```python
def __init__(name: Text, silent_fail: Optional[bool] = False) -> None
```

Creates action. See docstring of parent class.

#### intent\_name\_from\_action

```python
@staticmethod
def intent_name_from_action(action_name: Text) -> Text
```

Resolve the name of the intent from the action name.

#### get\_full\_retrieval\_name

```python
def get_full_retrieval_name(tracker: "DialogueStateTracker") -> Optional[Text]
```

Returns full retrieval name for the action.

Extracts retrieval intent from response selector and
returns complete action utterance name.

**Arguments**:

- `tracker` - Tracker containing past conversation events.
  

**Returns**:

  Full retrieval name of the action if the last user utterance
  contains a response selector output, `None` otherwise.

#### run

```python
async def run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Query the appropriate response and create a bot utterance with that.

#### name

```python
def name() -> Text
```

Returns action name.

## ActionBack Objects

```python
class ActionBack(ActionBotResponse)
```

Revert the tracker state by two user utterances.

#### name

```python
def name() -> Text
```

Returns action back name.

#### \_\_init\_\_

```python
def __init__() -> None
```

Initializes action back.

#### run

```python
async def run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Runs action. Please see parent class for the full docstring.

## ActionListen Objects

```python
class ActionListen(Action)
```

The first action in any turn - bot waits for a user message.

The bot should stop taking further actions and wait for the user to say
something.

#### run

```python
async def run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Runs action. Please see parent class for the full docstring.

## ActionRestart Objects

```python
class ActionRestart(ActionBotResponse)
```

Resets the tracker to its initial state.

Utters the restart response if available.

#### name

```python
def name() -> Text
```

Returns action restart name.

#### \_\_init\_\_

```python
def __init__() -> None
```

Initializes action restart.

#### run

```python
async def run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Runs action. Please see parent class for the full docstring.

## ActionSessionStart Objects

```python
class ActionSessionStart(Action)
```

Applies a conversation session start.

Takes all `SlotSet` events from the previous session and applies them to the new
session.

#### run

```python
async def run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Runs action. Please see parent class for the full docstring.

## ActionDefaultFallback Objects

```python
class ActionDefaultFallback(ActionBotResponse)
```

Executes the fallback action and goes back to the prev state of the dialogue.

#### name

```python
def name() -> Text
```

Returns action default fallback name.

#### \_\_init\_\_

```python
def __init__() -> None
```

Initializes action default fallback.

#### run

```python
async def run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Runs action. Please see parent class for the full docstring.

## ActionDeactivateLoop Objects

```python
class ActionDeactivateLoop(Action)
```

Deactivates an active loop.

#### run

```python
async def run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Runs action. Please see parent class for the full docstring.

## RemoteAction Objects

```python
class RemoteAction(Action)
```

#### action\_response\_format\_spec

```python
@staticmethod
def action_response_format_spec() -> Dict[Text, Any]
```

Expected response schema for an Action endpoint.

Used for validation of the response returned from the
Action endpoint.

#### run

```python
async def run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Runs action. Please see parent class for the full docstring.

## ActionExecutionRejection Objects

```python
class ActionExecutionRejection(RasaException)
```

Raising this exception will allow other policies
to predict a different action

## ActionRevertFallbackEvents Objects

```python
class ActionRevertFallbackEvents(Action)
```

Reverts events which were done during the `TwoStageFallbackPolicy`.

This reverts user messages and bot utterances done during a fallback
of the `TwoStageFallbackPolicy`. By doing so it is not necessary to
write custom stories for the different paths, but only of the happy
path. This is deprecated and can be removed once the
`TwoStageFallbackPolicy` is removed.

#### run

```python
async def run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Runs action. Please see parent class for the full docstring.

## ActionUnlikelyIntent Objects

```python
class ActionUnlikelyIntent(Action)
```

An action that indicates that the intent predicted by NLU is unexpected.

This action can be predicted by `UnexpecTEDIntentPolicy`.

#### name

```python
def name() -> Text
```

Returns the name of the action.

#### run

```python
async def run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Runs action. Please see parent class for the full docstring.

#### has\_user\_affirmed

```python
def has_user_affirmed(tracker: "DialogueStateTracker") -> bool
```

Indicates if the last executed action is `action_default_ask_affirmation`.

## ActionDefaultAskAffirmation Objects

```python
class ActionDefaultAskAffirmation(Action)
```

Default implementation which asks the user to affirm his intent.

It is suggested to overwrite this default action with a custom action
to have more meaningful prompts for the affirmations. E.g. have a
description of the intent instead of its identifier name.

#### run

```python
async def run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Runs action. Please see parent class for the full docstring.

## ActionDefaultAskRephrase Objects

```python
class ActionDefaultAskRephrase(ActionBotResponse)
```

Default implementation which asks the user to rephrase his intent.

#### name

```python
def name() -> Text
```

Returns action default ask rephrase name.

#### \_\_init\_\_

```python
def __init__() -> None
```

Initializes action default ask rephrase.

