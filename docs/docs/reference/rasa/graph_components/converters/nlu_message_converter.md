---
sidebar_label: rasa.graph_components.converters.nlu_message_converter
title: rasa.graph_components.converters.nlu_message_converter
---
## NLUMessageConverter Objects

```python
class NLUMessageConverter(GraphComponent)
```

Converts the user message into a NLU Message object.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> NLUMessageConverter
```

Creates component (see parent class for full docstring).

#### convert\_user\_message

```python
 | convert_user_message(messages: List[UserMessage]) -> List[Message]
```

Converts user message into Message object.

**Arguments**:

- `messages` - The user messages which should be converted to be processed by
  the NLU components.
  

**Returns**:

  List containing only one instance of Message.
  Else empty list if user message is None.

