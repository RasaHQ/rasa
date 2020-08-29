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
 | get_sparse_features(attribute: Text, featurizers: Optional[List[Text]] = None) -> Tuple[Optional[scipy.sparse.spmatrix], Optional[scipy.sparse.spmatrix]]
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
 | get_dense_features(attribute: Text, featurizers: Optional[List[Text]] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]
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

