---
sidebar_label: train_utils
title: rasa.utils.train_utils
---

#### normalize

```python
normalize(values: np.ndarray, ranking_length: Optional[int] = 0) -> np.ndarray
```

Normalizes an array of positive numbers over the top `ranking_length` values.
Other values will be set to 0.

#### update\_similarity\_type

```python
update_similarity_type(config: Dict[Text, Any]) -> Dict[Text, Any]
```

If SIMILARITY_TYPE is set to &#x27;auto&#x27;, update the SIMILARITY_TYPE depending
on the LOSS_TYPE.

**Arguments**:

- `config` - model configuration
  
- `Returns` - updated model configuration

#### align\_token\_features

```python
align_token_features(list_of_tokens: List[List[Token]], in_token_features: np.ndarray, shape: Optional[Tuple] = None) -> np.ndarray
```

Align token features to match tokens.

ConveRTTokenizer, LanguageModelTokenizers might split up tokens into sub-tokens.
We need to take the mean of the sub-token vectors and take that as token vector.

**Arguments**:

- `list_of_tokens` - tokens for examples
- `in_token_features` - token features from ConveRT
- `shape` - shape of feature matrix
  

**Returns**:

  Token features.

#### update\_evaluation\_parameters

```python
update_evaluation_parameters(config: Dict[Text, Any]) -> Dict[Text, Any]
```

If EVAL_NUM_EPOCHS is set to -1, evaluate at the end of the training.

**Arguments**:

- `config` - model configuration
  
- `Returns` - updated model configuration

#### load\_tf\_hub\_model

```python
load_tf_hub_model(model_url: Text) -> Any
```

Load model from cache if possible, otherwise from TFHub

#### check\_deprecated\_options

```python
check_deprecated_options(config: Dict[Text, Any]) -> Dict[Text, Any]
```

If old model configuration parameters are present in the provided config, replace
them with the new parameters and log a warning.

**Arguments**:

- `config` - model configuration
  
- `Returns` - updated model configuration

