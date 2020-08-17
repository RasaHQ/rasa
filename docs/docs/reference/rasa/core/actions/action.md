---
sidebar_label: rasa.core.actions.action
title: rasa.core.actions.action
---

#### default\_actions

```python
default_actions(action_endpoint: Optional[EndpointConfig] = None) -> List["Action"]
```

List default actions.

#### default\_action\_names

```python
default_action_names() -> List[Text]
```

List default action names.

#### combine\_with\_templates

```python
combine_with_templates(actions: List[Text], templates: Dict[Text, Any]) -> List[Text]
```

Combines actions with utter actions listed in responses section.

#### action\_from\_name

```python
action_from_name(name: Text, action_endpoint: Optional[EndpointConfig], user_actions: List[Text], should_use_form_action: bool = False) -> "Action"
```

Return an action instance for the name.

#### actions\_from\_names

```python
actions_from_names(action_names: List[Text], action_endpoint: Optional[EndpointConfig], user_actions: List[Text]) -> List["Action"]
```

Converts the names of actions into class instances.

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

## ActionRetrieveResponse Objects

```python
class ActionRetrieveResponse(Action)
```

An action which queries the Response Selector for the appropriate response.

#### run

```python
 | async run(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain")
```

Query the appropriate response and create a bot utterance with that.

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

## ActionDeactivateForm Objects

```python
class ActionDeactivateForm(Action)
```

Deactivates a form

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
class ActionExecutionRejection(Exception)
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
path.

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

