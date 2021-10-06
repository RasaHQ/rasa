---
sidebar_label: rasa.shared.nlu.interpreter
title: rasa.shared.nlu.interpreter
---
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
 | async parse(text: Text, message_id: Optional[Text] = None, tracker: Optional[DialogueStateTracker] = None, metadata: Optional[Dict] = None) -> Dict[Text, Any]
```

Parse a text message.

#### synchronous\_parse

```python
 | synchronous_parse(text: Text) -> Dict[Text, Any]
```

Parse a text message.

