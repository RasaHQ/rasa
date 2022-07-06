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

#### blueprint

```python
def blueprint(on_new_message: Callable[[UserMessage], Awaitable[None]]) -> Blueprint
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
@classmethod
def name(cls) -> Text
```

Name of QueueOutputChannel.

