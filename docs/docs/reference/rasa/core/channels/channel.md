---
sidebar_label: rasa.core.channels.channel
title: rasa.core.channels.channel
---
## UserMessage Objects

```python
class UserMessage()
```

Represents an incoming message.

Includes the channel the responses should be sent to.

#### \_\_init\_\_

```python
 | __init__(text: Optional[Text] = None, output_channel: Optional["OutputChannel"] = None, sender_id: Optional[Text] = None, parse_data: Dict[Text, Any] = None, input_channel: Optional[Text] = None, message_id: Optional[Text] = None, metadata: Optional[Dict] = None) -> None
```

Creates a ``UserMessage`` object.

**Arguments**:

- `text` - the message text content.
- `output_channel` - the output channel which should be used to send
  bot responses back to the user.
- `sender_id` - the message owner ID.
- `parse_data` - rasa data about the message.
- `input_channel` - the name of the channel which received this message.
- `message_id` - ID of the message.
- `metadata` - additional metadata for this message.

#### register

```python
register(input_channels: List["InputChannel"], app: Sanic, route: Optional[Text]) -> None
```

Registers input channel blueprints with Sanic.

## InputChannel Objects

```python
class InputChannel()
```

Input channel base class.

#### name

```python
 | @classmethod
 | name(cls) -> Text
```

Every input channel needs a name to identify it.

#### blueprint

```python
 | blueprint(on_new_message: Callable[[UserMessage], Awaitable[Any]]) -> Blueprint
```

Defines a Sanic blueprint.

The blueprint will be attached to a running sanic server and handle
incoming routes it registered for.

#### get\_output\_channel

```python
 | get_output_channel() -> Optional["OutputChannel"]
```

Create ``OutputChannel`` based on information provided by the input channel.

Implementing this function is not required. If this function returns a valid
``OutputChannel`` this can be used by Rasa to send bot responses to the user
without the user initiating an interaction.

**Returns**:

  ``OutputChannel`` instance or ``None`` in case creating an output channel
  only based on the information present in the ``InputChannel`` is not
  possible.

#### get\_metadata

```python
 | get_metadata(request: Request) -> Optional[Dict[Text, Any]]
```

Extracts additional information from the incoming request.

Implementing this function is not required. However, it can be used to extract
metadata from the request. The return value is passed on to the
``UserMessage`` object and stored in the conversation tracker.

**Arguments**:

- `request` - incoming request with the message of the user
  

**Returns**:

  Metadata which was extracted from the request.

#### decode\_jwt

```python
decode_jwt(bearer_token: Text, jwt_key: Text, jwt_algorithm: Text) -> Dict
```

Decodes a Bearer Token using the specific JWT key and algorithm.

**Arguments**:

- `bearer_token` - Encoded Bearer token
- `jwt_key` - Public JWT key for decoding the Bearer token
- `jwt_algorithm` - JWT algorithm used for decoding the Bearer token
  

**Returns**:

  `Dict` containing the decoded payload if successful or an exception
  if unsuccessful

#### decode\_bearer\_token

```python
decode_bearer_token(bearer_token: Text, jwt_key: Text, jwt_algorithm: Text) -> Optional[Dict]
```

Decodes a Bearer Token using the specific JWT key and algorithm.

**Arguments**:

- `bearer_token` - Encoded Bearer token
- `jwt_key` - Public JWT key for decoding the Bearer token
- `jwt_algorithm` - JWT algorithm used for decoding the Bearer token
  

**Returns**:

  `Dict` containing the decoded payload if successful or `None` if unsuccessful

## OutputChannel Objects

```python
class OutputChannel()
```

Output channel base class.

Provides sane implementation of the send methods
for text only output channels.

#### name

```python
 | @classmethod
 | name(cls) -> Text
```

Every output channel needs a name to identify it.

#### send\_response

```python
 | async send_response(recipient_id: Text, message: Dict[Text, Any]) -> None
```

Send a message to the client.

#### send\_text\_message

```python
 | async send_text_message(recipient_id: Text, text: Text, **kwargs: Any) -> None
```

Send a message through this channel.

#### send\_image\_url

```python
 | async send_image_url(recipient_id: Text, image: Text, **kwargs: Any) -> None
```

Sends an image. Default will just post the url as a string.

#### send\_attachment

```python
 | async send_attachment(recipient_id: Text, attachment: Text, **kwargs: Any) -> None
```

Sends an attachment. Default will just post as a string.

#### send\_text\_with\_buttons

```python
 | async send_text_with_buttons(recipient_id: Text, text: Text, buttons: List[Dict[Text, Any]], **kwargs: Any, ,) -> None
```

Sends buttons to the output.

Default implementation will just post the buttons as a string.

#### send\_quick\_replies

```python
 | async send_quick_replies(recipient_id: Text, text: Text, quick_replies: List[Dict[Text, Any]], **kwargs: Any, ,) -> None
```

Sends quick replies to the output.

Default implementation will just send as buttons.

#### send\_elements

```python
 | async send_elements(recipient_id: Text, elements: Iterable[Dict[Text, Any]], **kwargs: Any) -> None
```

Sends elements to the output.

Default implementation will just post the elements as a string.

#### send\_custom\_json

```python
 | async send_custom_json(recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any) -> None
```

Sends json dict to the output channel.

Default implementation will just post the json contents as a string.

## CollectingOutputChannel Objects

```python
class CollectingOutputChannel(OutputChannel)
```

Output channel that collects send messages in a list

(doesn&#x27;t send them anywhere, just collects them).

#### send\_image\_url

```python
 | async send_image_url(recipient_id: Text, image: Text, **kwargs: Any) -> None
```

Sends an image. Default will just post the url as a string.

#### send\_attachment

```python
 | async send_attachment(recipient_id: Text, attachment: Text, **kwargs: Any) -> None
```

Sends an attachment. Default will just post as a string.

