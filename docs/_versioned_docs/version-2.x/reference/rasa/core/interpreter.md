---
sidebar_label: rasa.core.interpreter
title: rasa.core.interpreter
---
#### create\_interpreter

```python
create_interpreter(obj: Union[
        rasa.shared.nlu.interpreter.NaturalLanguageInterpreter,
        EndpointConfig,
        Text,
        None,
    ]) -> "rasa.shared.nlu.interpreter.NaturalLanguageInterpreter"
```

Factory to create a natural language interpreter.

## RasaNLUHttpInterpreter Objects

```python
class RasaNLUHttpInterpreter(rasa.shared.nlu.interpreter.NaturalLanguageInterpreter)
```

#### parse

```python
 | async parse(text: Text, message_id: Optional[Text] = None, tracker: Optional[DialogueStateTracker] = None, metadata: Optional[Dict] = None) -> Dict[Text, Any]
```

Parse a text message.

Return a default value if the parsing of the text failed.

## RasaNLUInterpreter Objects

```python
class RasaNLUInterpreter(rasa.shared.nlu.interpreter.NaturalLanguageInterpreter)
```

#### parse

```python
 | async parse(text: Text, message_id: Optional[Text] = None, tracker: Optional[DialogueStateTracker] = None, metadata: Optional[Dict] = None) -> Dict[Text, Any]
```

Parse a text message.

Return a default value if the parsing of the text failed.

#### featurize\_message

```python
 | featurize_message(message: Message) -> Optional[Message]
```

Featurize message using a trained NLU pipeline.

**Arguments**:

- `message` - storing text to process

**Returns**:

  message containing tokens and features which are the output of the NLU
  pipeline

