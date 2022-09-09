---
sidebar_label: rasa.core.channels.rest
title: rasa.core.channels.rest
---
## RestInput Objects

```python
class RestInput(InputChannel)
```

A custom http input channel.

This implementation is the basis for a custom implementation of a chat
frontend. You can customize this to send messages to Rasa and
retrieve responses from the assistant.

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

#### stream\_response

```python
 | stream_response(on_new_message: Callable[[UserMessage], Awaitable[None]], text: Text, sender_id: Text, input_channel: Text, metadata: Optional[Dict[Text, Any]]) -> Callable[[Any], Awaitable[None]]
```

Streams response to the client.

If the stream option is enabled, this method will be called to
stream the response to the client

**Arguments**:

- `on_new_message` - sanic event
- `text` - message text
- `sender_id` - message sender_id
- `input_channel` - input channel name
- `metadata` - optional metadata sent with the message
  

**Returns**:

  Sanic stream

#### blueprint

```python
 | blueprint(on_new_message: Callable[[UserMessage], Awaitable[None]]) -> Blueprint
```

Groups the collection of endpoints used by rest channel.

## QueueOutputChannel Objects

```python
class QueueOutputChannel(CollectingOutputChannel)
```

Output channel that collects send messages in a list

(doesn&#x27;t send them anywhere, just collects them).

#### name

```python
 | @classmethod
 | name(cls) -> Text
```

Name of QueueOutputChannel.

