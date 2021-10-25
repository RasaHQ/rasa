---
sidebar_label: rasa.nlu.classifiers.regex_message_handler
title: rasa.nlu.classifiers.regex_message_handler
---
## RegexMessageHandler Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=False
)
class RegexMessageHandler(GraphComponent,  EntityExtractorMixin)
```

Handles hardcoded NLU predictions from messages starting with a `/`.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> RegexMessageHandler
```

Creates a new untrained component (see parent class for full docstring).

#### process

```python
def process(messages: List[Message], domain: Optional[Domain] = None) -> List[Message]
```

Adds hardcoded intents and entities for messages starting with &#x27;/&#x27;.

**Arguments**:

- `messages` - The messages which should be handled.
- `domain` - If given the domain is used to check whether the intent, entities
  valid.
  

**Returns**:

  The messages with potentially intent and entity prediction replaced
  in case the message started with a `/`.

