---
sidebar_label: rasa.core.exceptions
title: rasa.core.exceptions
---

## RasaCoreException Objects

```python
class RasaCoreException(RasaException)
```

Basic exception for errors raised by Rasa Core.

## StoryParseError Objects

```python
class StoryParseError(RasaCoreException,  ValueError)
```

Raised if there is an error while parsing a story file.

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

