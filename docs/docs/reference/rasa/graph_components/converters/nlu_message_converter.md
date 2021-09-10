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
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> NLUMessageConverter
```

Creates component (see parent class for full docstring).

#### convert\_user\_message

```python
@staticmethod
def convert_user_message(message: Optional[UserMessage]) -> List[Message]
```

Converts user message into Message object.

**Returns**:

  List containing only one instance of Message.
  Else empty list if user message is None.

