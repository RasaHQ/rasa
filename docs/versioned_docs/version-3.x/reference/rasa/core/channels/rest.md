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

## QueueOutputChannel Objects

```python
class QueueOutputChannel(CollectingOutputChannel)
```

Output channel that collects send messages in a list

(doesn&#x27;t send them anywhere, just collects them).

