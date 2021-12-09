---
sidebar_label: rasa.nlu.training_data.message
title: rasa.nlu.training_data.message
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

#### build

```python
 | @classmethod
 | build(cls, text: Text, intent: Optional[Text] = None, entities: List[Dict[Text, Any]] = None, **kwargs: Any, ,) -> "Message"
```

Build a Message from `UserUttered` data.

**Arguments**:

- `text` - text of a user&#x27;s utterance
- `intent` - an intent of the user utterance
- `entities` - entities in the user&#x27;s utterance

**Returns**:

  Message

#### build\_from\_action

```python
 | @classmethod
 | build_from_action(cls, action_text: Optional[Text] = "", action_name: Optional[Text] = "", **kwargs: Any, ,) -> "Message"
```

Build a `Message` from `ActionExecuted` data.

**Arguments**:

- `action_text` - text of a bot&#x27;s utterance
- `action_name` - name of an action executed

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

Get all sparse features for the given attribute that are coming from the
given list of featurizers.
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

Get all dense features for the given attribute that are coming from the given
list of featurizers.
If no featurizers are provided, all available features will be considered.

**Arguments**:

- `attribute` - message attribute
- `featurizers` - names of featurizers to consider

**Returns**:

  Dense features.

#### features\_present

```python
 | features_present(attribute: Text, featurizers: Optional[List[Text]] = None) -> bool
```

Check if there are any features present for the given attribute and
featurizers.
If no featurizers are provided, all available features will be considered.

**Arguments**:

- `attribute` - message attribute
- `featurizers` - names of featurizers to consider

**Returns**:

  ``True``, if features are present, ``False`` otherwise

