---
sidebar_label: rasa.core.training.interactive
title: rasa.core.training.interactive
---
## RestartConversation Objects

```python
class RestartConversation(Exception)
```

Exception used to break out the flow and restart the conversation.

## ForkTracker Objects

```python
class ForkTracker(Exception)
```

Exception used to break out the flow and fork at a previous step.

The tracker will be reset to the selected point in the past and the
conversation will continue from there.

## UndoLastStep Objects

```python
class UndoLastStep(Exception)
```

Exception used to break out the flow and undo the last step.

The last step is either the most recent user message or the most
recent action run by the bot.

## Abort Objects

```python
class Abort(Exception)
```

Exception used to abort the interactive learning and exit.

#### send\_message

```python
async send_message(endpoint: EndpointConfig, conversation_id: Text, message: Text, parse_data: Optional[Dict[Text, Any]] = None) -> Dict[Text, Any]
```

Send a user message to a conversation.

#### request\_prediction

```python
async request_prediction(endpoint: EndpointConfig, conversation_id: Text) -> Dict[Text, Any]
```

Request the next action prediction from core.

#### retrieve\_domain

```python
async retrieve_domain(endpoint: EndpointConfig) -> Dict[Text, Any]
```

Retrieve the domain from core.

#### retrieve\_status

```python
async retrieve_status(endpoint: EndpointConfig) -> Dict[Text, Any]
```

Retrieve the status from core.

#### retrieve\_tracker

```python
async retrieve_tracker(endpoint: EndpointConfig, conversation_id: Text, verbosity: EventVerbosity = EventVerbosity.ALL) -> Dict[Text, Any]
```

Retrieve a tracker from core.

#### send\_action

```python
async send_action(endpoint: EndpointConfig, conversation_id: Text, action_name: Text, policy: Optional[Text] = None, confidence: Optional[float] = None, is_new_action: bool = False) -> Dict[Text, Any]
```

Log an action to a conversation.

#### send\_event

```python
async send_event(endpoint: EndpointConfig, conversation_id: Text, evt: Union[List[Dict[Text, Any]], Dict[Text, Any]]) -> Dict[Text, Any]
```

Log an event to a conversation.

#### format\_bot\_output

```python
format_bot_output(message: BotUttered) -> Text
```

Format a bot response to be displayed in the history table.

#### latest\_user\_message

```python
latest_user_message(events: List[Dict[Text, Any]]) -> Optional[Dict[Text, Any]]
```

Return most recent user message.

#### is\_listening\_for\_message

```python
async is_listening_for_message(conversation_id: Text, endpoint: EndpointConfig) -> bool
```

Check if the conversation is in need for a user message.

#### record\_messages

```python
async record_messages(endpoint: EndpointConfig, file_importer: TrainingDataImporter, conversation_id: Text = DEFAULT_SENDER_ID, max_message_limit: Optional[int] = None, skip_visualization: bool = False) -> None
```

Read messages from the command line and print bot responses.

#### start\_visualization

```python
start_visualization(image_path: Text, port: int) -> None
```

Add routes to serve the conversation visualization files.

#### run\_interactive\_learning

```python
run_interactive_learning(file_importer: TrainingDataImporter, skip_visualization: bool = False, conversation_id: Text = uuid.uuid4().hex, server_args: Dict[Text, Any] = None) -> None
```

Start the interactive learning with the model of the agent.

#### calc\_true\_wrapping\_width

```python
calc_true_wrapping_width(text: Text, monospace_wrapping_width: int) -> int
```

Calculates a wrapping width that also works for CJK characters.

Chinese, Japanese and Korean characters are often broader than ascii
characters:
abcdefgh (8 chars)
我要去北京 (5 chars, roughly same visible width)

We need to account for that otherwise the wrapping doesn&#x27;t work
appropriately for long strings and the table overflows and creates
errors.

params:
    text: text sequence that should be wrapped into multiple lines
    monospace_wrapping_width: the maximum width per line in number of
        standard ascii characters
returns:
    The maximum line width for the given string that takes into account
    the strings visible width, so that it won&#x27;t lead to table overflow.

