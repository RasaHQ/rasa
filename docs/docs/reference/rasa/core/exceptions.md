---
sidebar_label: exceptions
title: rasa.core.exceptions
---

## UnsupportedDialogueModelError Objects

```python
class UnsupportedDialogueModelError(RasaCoreException)
```

Raised when a model is too old to be loaded.

**Attributes**:

- `message` - explanation of why the model is invalid

## AgentNotReady Objects

```python
class AgentNotReady(RasaCoreException)
```

Raised if someone tries to use an agent that is not ready.

An agent might be created, e.g. without an interpreter attached. But
if someone tries to parse a message with that agent, this exception
will be thrown.

