---
sidebar_label: rasa.nlu.classifiers.regex_message_handler
title: rasa.nlu.classifiers.regex_message_handler
---
## RegexMessageHandler Objects

```python
class RegexMessageHandler(GraphComponent)
```

Unpacks messages where `TEXT` contains an encoding of attributes.

The `TEXT` attribute of such messages consists of the following sub-strings:
1. special symbol &quot;/&quot; (mandatory)
2. intent name (mandatory)
3. &quot;@&lt;confidence value&gt;&quot; where the value can be any int or float (optional)
4. string representation of a dictionary mapping entity types to entity
   values (optional)

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> RegexMessageHandler
```

Creates a new untrained component (see parent class for full docstring).

#### \_\_init\_\_

```python
def __init__() -> None
```

Creates a new instance.

#### process

```python
def process(messages: List[Message], domain: Domain) -> List[Message]
```

Unpacks messages where `TEXT` contains an encoding of attributes.

Note that this method returns a *new* message instance if there is
something to unpack in the given message (and returns the given message
otherwise). The new message is created on purpose to get rid of all attributes
that NLU components might have added based on the `TEXT` attribute which
does not contain real text but the regex we expect here.

**Arguments**:

- `messages` - list of messages
- `domain` - the domain

**Returns**:

  list of messages where the i-th message is equal to the i-th input message
  if that message does not need to be unpacked, and a new message with the
  extracted attributes otherwise

