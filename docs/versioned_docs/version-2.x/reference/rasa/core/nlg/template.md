---
sidebar_label: rasa.core.nlg.template
title: rasa.core.nlg.template
---
## TemplatedNaturalLanguageGenerator Objects

```python
class TemplatedNaturalLanguageGenerator(NaturalLanguageGenerator)
```

Natural language generator that generates messages based on templates.

The templates can use variables to customize the utterances based on the
state of the dialogue.

#### generate

```python
 | async generate(template_name: Text, tracker: DialogueStateTracker, output_channel: Text, **kwargs: Any, ,) -> Optional[Dict[Text, Any]]
```

Generate a response for the requested template.

#### generate\_from\_slots

```python
 | generate_from_slots(template_name: Text, filled_slots: Dict[Text, Any], output_channel: Text, **kwargs: Any, ,) -> Optional[Dict[Text, Any]]
```

Generate a response for the requested template.

