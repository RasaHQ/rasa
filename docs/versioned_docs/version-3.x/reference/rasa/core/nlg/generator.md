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
async def generate(utter_action: Text, tracker: "DialogueStateTracker",
                   output_channel: Text,
                   **kwargs: Any) -> Optional[Dict[Text, Any]]
```

Generate a response for the requested utter action.

There are a lot of different methods to implement this, e.g. the
generation can be based on responses or be fully ML based by feeding
the dialogue state into a machine learning NLG model.

#### create

```python
@staticmethod
def create(obj: Union["NaturalLanguageGenerator", EndpointConfig, None],
           domain: Optional[Domain]) -> "NaturalLanguageGenerator"
```

Factory to create a generator.

## ResponseVariationFilter Objects

```python
class ResponseVariationFilter()
```

Filters response variations based on the channel, action and condition.

#### responses\_for\_utter\_action

```python
def responses_for_utter_action(
        utter_action: Text, output_channel: Text,
        filled_slots: Dict[Text, Any]) -> List[Dict[Text, Any]]
```

Returns array of responses that fit the channel, action and condition.

#### get\_response\_variation\_id

```python
def get_response_variation_id(utter_action: Text,
                              tracker: DialogueStateTracker,
                              output_channel: Text) -> Optional[Text]
```

Returns the first matched response variation ID.

This ID corresponds to the response variation that fits
the channel, action and condition.

