---
sidebar_label: message
title: rasa.shared.nlu.training_data.message
---

## Message Objects

```python
class Message()
```

#### as\_dict\_nlu

```python
 | as_dict_nlu() -> dict
```

Get dict representation of message as it would appear in training data

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

#### get\_combined\_intent\_response\_key

```python
 | get_combined_intent_response_key() -> Text
```

Get intent as it appears in training data

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

