---
sidebar_label: rasa.core.nlg.generator
title: rasa.core.nlg.generator
---
## NaturalLanguageGenerator Objects

```python
class NaturalLanguageGenerator()
```

Generate bot utterances based on a dialogue state.

#### generate

```python
 | async generate(utter_action: Text, tracker: "DialogueStateTracker", output_channel: Text, **kwargs: Any, ,) -> Optional[Dict[Text, Any]]
```

Generate a response for the requested utter action.

There are a lot of different methods to implement this, e.g. the
generation can be based on responses or be fully ML based by feeding
the dialogue state into a machine learning NLG model.

#### create

```python
 | @staticmethod
 | create(obj: Union["NaturalLanguageGenerator", EndpointConfig, None], domain: Optional[Domain]) -> "NaturalLanguageGenerator"
```

Factory to create a generator.

