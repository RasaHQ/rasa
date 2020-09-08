---
sidebar_label: rasa.core.interpreter
title: rasa.core.interpreter
---

## NaturalLanguageInterpreter Objects

```python
class NaturalLanguageInterpreter()
```

#### create

```python
 | @staticmethod
 | create(obj: Union["NaturalLanguageInterpreter", EndpointConfig, Text, None]) -> "NaturalLanguageInterpreter"
```

Factory to create an natural language interpreter.

## RegexInterpreter Objects

```python
class RegexInterpreter(NaturalLanguageInterpreter)
```

#### extract\_intent\_and\_entities

```python
 | @staticmethod
 | extract_intent_and_entities(user_input: Text) -> Tuple[Optional[Text], float, List[Dict[Text, Any]]]
```

Parse the user input using regexes to extract intent &amp; entities.

#### parse

```python
 | async parse(text: Text, message_id: Optional[Text] = None, tracker: Optional[DialogueStateTracker] = None) -> Dict[Text, Any]
```

Parse a text message.

#### synchronous\_parse

```python
 | synchronous_parse(text: Text) -> Dict[Text, Any]
```

Parse a text message.

## RasaNLUHttpInterpreter Objects

```python
class RasaNLUHttpInterpreter(NaturalLanguageInterpreter)
```

#### parse

```python
 | async parse(text: Text, message_id: Optional[Text] = None, tracker: Optional[DialogueStateTracker] = None) -> Dict[Text, Any]
```

Parse a text message.

Return a default value if the parsing of the text failed.

## RasaNLUInterpreter Objects

```python
class RasaNLUInterpreter(NaturalLanguageInterpreter)
```

#### parse

```python
 | async parse(text: Text, message_id: Optional[Text] = None, tracker: Optional[DialogueStateTracker] = None) -> Dict[Text, Any]
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

  message containing tokens and features which are the output of the NLU pipeline

