---
sidebar_label: rasa.core.nlg.response
title: rasa.core.nlg.response
---
## TemplatedNaturalLanguageGenerator Objects

```python
class TemplatedNaturalLanguageGenerator(NaturalLanguageGenerator)
```

Natural language generator that generates messages based on responses.

The responses can use variables to customize the utterances based on the
state of the dialogue.

#### \_\_init\_\_

```python
def __init__(responses: Dict[Text, List[Dict[Text, Any]]]) -> None
```

Creates a Template Natural Language Generator.

**Arguments**:

- `responses` - responses that will be used to generate messages.

#### generate

```python
async def generate(utter_action: Text, tracker: DialogueStateTracker, output_channel: Text, **kwargs: Any, ,) -> Optional[Dict[Text, Any]]
```

Generate a response for the requested utter action.

#### generate\_from\_slots

```python
def generate_from_slots(utter_action: Text, filled_slots: Dict[Text, Any], output_channel: Text, **kwargs: Any, ,) -> Optional[Dict[Text, Any]]
```

Generate a response for the requested utter action.

