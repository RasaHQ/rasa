---
sidebar_label: rasa.shared.nlu.training_data.message
title: rasa.shared.nlu.training_data.message
---
## Message Objects

```python
class Message()
```

Container for data that can be used to describe a conversation turn.

The turn is described by a set of attributes such as e.g. `TEXT`  and  `INTENT`
when describing a user utterance or e.g. `ACTION_NAME` for describing a bot action.
The container includes raw information (`self.data`) as well as features
(`self.features`) for each such attribute.
Moreover, the message has a timestamp and can keep track about information
on a specific subset of attributes (`self.output_properties`).

#### \_\_init\_\_

```python
 | __init__(data: Optional[Dict[Text, Any]] = None, output_properties: Optional[Set] = None, time: Optional[int] = None, features: Optional[List["Features"]] = None, **kwargs: Any, ,) -> None
```

Creates an instance of Message.

#### add\_diagnostic\_data

```python
 | add_diagnostic_data(origin: Text, data: Dict[Text, Any]) -> None
```

Adds diagnostic data from the `origin` component.

**Arguments**:

- `origin` - Name of the component that created the data.
- `data` - The diagnostic data.

#### set

```python
 | set(prop: Text, info: Any, add_to_output: bool = False) -> None
```

Sets the message&#x27;s property to the given value.

**Arguments**:

- `prop` - Name of the property to be set.
- `info` - Value to be assigned to that property.
- `add_to_output` - Decides whether to add `prop` to the `output_properties`.

#### as\_dict\_nlu

```python
 | as_dict_nlu() -> dict
```

Get dict representation of message as it would appear in training data

#### as\_dict

```python
 | as_dict(only_output_properties: bool = False) -> Dict
```

Gets dict representation of message.

#### \_\_hash\_\_

```python
 | __hash__() -> int
```

Calculate a hash for the message.

**Returns**:

  Hash of the message.

#### fingerprint

```python
 | fingerprint() -> Text
```

Calculate a string fingerprint for the message.

**Returns**:

  Fingerprint of the message.

#### build

```python
 | @classmethod
 | build(cls, text: Text, intent: Optional[Text] = None, entities: Optional[List[Dict[Text, Any]]] = None, intent_metadata: Optional[Any] = None, example_metadata: Optional[Any] = None, **kwargs: Any, ,) -> "Message"
```

Builds a Message from `UserUttered` data.

**Arguments**:

- `text` - text of a user&#x27;s utterance
- `intent` - an intent of the user utterance
- `entities` - entities in the user&#x27;s utterance
- `intent_metadata` - optional metadata for the intent
- `example_metadata` - optional metadata for the intent example
  

**Returns**:

  Message

#### get\_full\_intent

```python
 | get_full_intent() -> Text
```

Get intent as it appears in training data

#### separate\_intent\_response\_key

```python
 | @staticmethod
 | separate_intent_response_key(original_intent: Text) -> Tuple[Text, Optional[Text]]
```

Splits intent into main intent name and optional sub-intent name.

For example, `&quot;FAQ/how_to_contribute&quot;` would be split into
`(&quot;FAQ&quot;, &quot;how_to_contribute&quot;)`. The response delimiter can
take different values (not just `&quot;/&quot;`) and depends on the
constant - `RESPONSE_IDENTIFIER_DELIMITER`.
If there is no response delimiter in the intent, the second tuple
item is `None`, e.g. `&quot;FAQ&quot;` would be mapped to `(&quot;FAQ&quot;, None)`.

#### get\_sparse\_features

```python
 | get_sparse_features(attribute: Text, featurizers: Optional[List[Text]] = None) -> Tuple[Optional["Features"], Optional["Features"]]
```

Gets all sparse features for the attribute given the list of featurizers.

If no featurizers are provided, all available features will be considered.

**Arguments**:

- `attribute` - message attribute
- `featurizers` - names of featurizers to consider
  

**Returns**:

  Sparse features.

#### get\_sparse\_feature\_sizes

```python
 | get_sparse_feature_sizes(attribute: Text, featurizers: Optional[List[Text]] = None) -> Dict[Text, List[int]]
```

Gets sparse feature sizes for the attribute given the list of featurizers.

If no featurizers are provided, all available features will be considered.

**Arguments**:

- `attribute` - message attribute
- `featurizers` - names of featurizers to consider
  

**Returns**:

  Sparse feature sizes.

#### get\_dense\_features

```python
 | get_dense_features(attribute: Text, featurizers: Optional[List[Text]] = None) -> Tuple[Optional["Features"], Optional["Features"]]
```

Gets all dense features for the attribute given the list of featurizers.

If no featurizers are provided, all available features will be considered.

**Arguments**:

- `attribute` - message attribute
- `featurizers` - names of featurizers to consider
  

**Returns**:

  Dense features.

#### get\_all\_features

```python
 | get_all_features(attribute: Text, featurizers: Optional[List[Text]] = None) -> List["Features"]
```

Gets all features for the attribute given the list of featurizers.

If no featurizers are provided, all available features will be considered.

**Arguments**:

- `attribute` - message attribute
- `featurizers` - names of featurizers to consider
  

**Returns**:

  Features.

#### features\_present

```python
 | features_present(attribute: Text, featurizers: Optional[List[Text]] = None) -> bool
```

Checks if there are any features present for the attribute and featurizers.

If no featurizers are provided, all available features will be considered.

**Arguments**:

- `attribute` - Message attribute.
- `featurizers` - Names of featurizers to consider.
  

**Returns**:

  ``True``, if features are present, ``False`` otherwise.

#### is\_core\_or\_domain\_message

```python
 | is_core_or_domain_message() -> bool
```

Checks whether the message is a core message or from the domain.

E.g. a core message is created from a story or a domain action,
not from the NLU data.

**Returns**:

  True, if message is a core or domain message, false otherwise.

#### is\_e2e\_message

```python
 | is_e2e_message() -> bool
```

Checks whether the message came from an e2e story.

**Returns**:

  `True`, if message is a from an e2e story, `False` otherwise.

#### find\_overlapping\_entities

```python
 | find_overlapping_entities() -> List[Tuple[Dict[Text, Any], Dict[Text, Any]]]
```

Finds any overlapping entity annotations.

