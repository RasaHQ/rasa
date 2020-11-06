---
sidebar_label: rasa.core.actions.action
title: rasa.core.actions.action
---

#### default\_actions

```python
default_actions(action_endpoint: Optional[EndpointConfig] = None) -> List["Action"]
```

List default actions.

#### action\_for\_index

```python
action_for_index(index: int, domain: Domain, action_endpoint: Optional[EndpointConfig]) -> "Action"
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

#### action\_for\_name

```python
action_for_name(action_name: Text, domain: Domain, action_endpoint: Optional[EndpointConfig]) -> "Action"
```

Create an `Action` object based on the name of the `Action`.

**Arguments**:

- `action_name` - The name of the `Action`.
- `domain` - The `Domain` of the current model. The domain contains the actions
  provided by the user + the default actions.
- `action_endpoint` - Can be used to run `custom_actions`
  (e.g. using the `rasa-sdk`).
  

**Returns**:

  The instantiated `Action` or `None` if no `Action` was found for the given
  index.

#### is\_retrieval\_action

```python
is_retrieval_action(action_name: Text, retrieval_intents: List[Text]) -> bool
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

#### action\_from\_name

```python
action_from_name(name: Text, domain: Domain, action_endpoint: Optional[EndpointConfig]) -> "Action"
```

Retrieves an action by its name.

**Arguments**:

- `name` - The name of the action.
- `domain` - The current model domain.
- `action_endpoint` - The endpoint to execute custom actions.
  

**Returns**:

  The instantiated action.

#### create\_bot\_utterance

```python
create_bot_utterance(message: Dict[Text, Any]) -> BotUttered
```

Create BotUttered event from message.

## Action Objects

```python
class Action()
```

Next action to be taken in response to a dialogue state.

#### name

```python
 | name() -> Text
```

Unique identifier of this simple action.

#### run

```python
 | async run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
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
- `metadata` - dictionary that can be sent to action server with custom
  data.

**Returns**:

- `List[Event]` - A list of :class:`rasa.core.events.Event` instances

## ActionUtterTemplate Objects

```python
class ActionUtterTemplate(Action)
```

An action which only effect is to utter a template when it is run.

Both, name and utter template, need to be specified using
the `name` method.

#### run

```python
 | async run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Simple run implementation uttering a (hopefully defined) template.

## ActionRetrieveResponse Objects

```python
class ActionRetrieveResponse(ActionUtterTemplate)
```

An action which queries the Response Selector for the appropriate response.

#### intent\_name\_from\_action

```python
 | @staticmethod
 | intent_name_from_action(action_name: Text) -> Text
```

Resolve the name of the intent from the action name.

#### action\_name\_from\_intent

```python
 | @staticmethod
 | action_name_from_intent(intent_name: Text) -> Text
```

Resolve the action name from the name of the intent.

#### run

```python
 | async run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Query the appropriate response and create a bot utterance with that.

## ActionBack Objects

```python
class ActionBack(ActionUtterTemplate)
```

Revert the tracker state by two user utterances.

## ActionListen Objects

```python
class ActionListen(Action)
```

The first action in any turn - bot waits for a user message.

The bot should stop taking further actions and wait for the user to say
something.

## ActionRestart Objects

```python
class ActionRestart(ActionUtterTemplate)
```

Resets the tracker to its initial state.

Utters the restart response if available.

## ActionSessionStart Objects

```python
class ActionSessionStart(Action)
```

Applies a conversation session start.

Takes all `SlotSet` events from the previous session and applies them to the new
session.

## ActionDefaultFallback Objects

```python
class ActionDefaultFallback(ActionUtterTemplate)
```

Executes the fallback action and goes back to the previous state
of the dialogue

## ActionDeactivateLoop Objects

```python
class ActionDeactivateLoop(Action)
```

Deactivates an active loop.

## RemoteAction Objects

```python
class RemoteAction(Action)
```

#### action\_response\_format\_spec

```python
 | @staticmethod
 | action_response_format_spec() -> Dict[Text, Any]
```

Expected response schema for an Action endpoint.

Used for validation of the response returned from the
Action endpoint.

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

## ActionDefaultAskAffirmation Objects

```python
class ActionDefaultAskAffirmation(Action)
```

Default implementation which asks the user to affirm his intent.

It is suggested to overwrite this default action with a custom action
to have more meaningful prompts for the affirmations. E.g. have a
description of the intent instead of its identifier name.

## ActionDefaultAskRephrase Objects

```python
class ActionDefaultAskRephrase(ActionUtterTemplate)
```

Default implementation which asks the user to rephrase his intent.

