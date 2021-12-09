---
sidebar_label: rasa.utils.train_utils
title: rasa.utils.train_utils
---
#### rank\_and\_mask

```python
rank_and_mask(confidences: np.ndarray, ranking_length: int = 0, renormalize: bool = False) -> Tuple[np.array, np.array]
```

Computes a ranking of the given confidences.

First, it computes a list containing the indices that would sort all the given
confidences in decreasing order.
If a `ranking_length` is specified, then only the indices for the `ranking_length`
largest confidences will be returned and all other confidences (i.e. whose indices
we do not return) will be masked by setting them to 0.
Moreover, if `renormalize` is set to `True`, then the confidences will
additionally be renormalised by dividing them by their sum.

We assume that the given confidences sum up to 1 and, if the
`ranking_length` is 0 or larger than the given number of confidences,
we set the `ranking_length` to the number of confidences.
Hence, in this case the confidences won&#x27;t be modified.

**Arguments**:

- `confidences` - a 1-d array of confidences that are non-negative and sum up to 1
- `ranking_length` - the size of the ranking to be computed. If set to 0 or
  something larger than the number of given confidences, then this is set
  to the exact number of given confidences.
- `renormalize` - determines whether the masked confidences should be renormalised.
  return_indices:

**Returns**:

  indices of the top `ranking_length` confidences and an array of the same
  shape as the given confidences that contains the possibly masked and
  renormalized confidence values

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
align_token_features(list_of_tokens: List[List["Token"]], in_token_features: np.ndarray, shape: Optional[Tuple] = None) -> np.ndarray
```

Align token features to match tokens.

ConveRTFeaturizer and LanguageModelFeaturizer might split up tokens into sub-tokens.
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

Update the config according to changed config params.

If old model configuration parameters are present in the provided config, replace
them with the new parameters and log a warning.

**Arguments**:

- `config` - model configuration
  
- `Returns` - updated model configuration

#### check\_core\_deprecated\_options

```python
check_core_deprecated_options(config: Dict[Text, Any]) -> Dict[Text, Any]
```

Update the core config according to changed config params.

If old model configuration parameters are present in the provided config, replace
them with the new parameters and log a warning.

**Arguments**:

- `config` - model configuration
  
- `Returns` - updated model configuration

#### entity\_label\_to\_tags

```python
entity_label_to_tags(model_predictions: Dict[Text, Any], entity_tag_specs: List["EntityTagSpec"], bilou_flag: bool = False, prediction_index: int = 0) -> Tuple[Dict[Text, List[Text]], Dict[Text, List[float]]]
```

Convert the output predictions for entities to the actual entity tags.

**Arguments**:

- `model_predictions` - the output predictions using the entity tag indices
- `entity_tag_specs` - the entity tag specifications
- `bilou_flag` - if &#x27;True&#x27;, the BILOU tagging schema was used
- `prediction_index` - the index in the batch of predictions
  to use for entity extraction
  

**Returns**:

  A map of entity tag type, e.g. entity, role, group, to actual entity tags and
  confidences.

#### create\_data\_generators

```python
create_data_generators(model_data: RasaModelData, batch_sizes: Union[int, List[int]], epochs: int, batch_strategy: Text = SEQUENCE, eval_num_examples: int = 0, random_seed: Optional[int] = None, shuffle: bool = True) -> Tuple[RasaBatchDataGenerator, Optional[RasaBatchDataGenerator]]
```

Create data generators for train and optional validation data.

**Arguments**:

- `model_data` - The model data to use.
- `batch_sizes` - The batch size(s).
- `epochs` - The number of epochs to train.
- `batch_strategy` - The batch strategy to use.
- `eval_num_examples` - Number of examples to use for validation data.
- `random_seed` - The random seed.
- `shuffle` - Whether to shuffle data inside the data generator.
  

**Returns**:

  The training data generator and optional validation data generator.

#### create\_common\_callbacks

```python
create_common_callbacks(epochs: int, tensorboard_log_dir: Optional[Text] = None, tensorboard_log_level: Optional[Text] = None, checkpoint_dir: Optional[Path] = None) -> List["Callback"]
```

Create common callbacks.

The following callbacks are created:
- RasaTrainingLogger callback
- Optional TensorBoard callback
- Optional RasaModelCheckpoint callback

**Arguments**:

- `epochs` - the number of epochs to train
- `tensorboard_log_dir` - optional directory that should be used for tensorboard
- `tensorboard_log_level` - defines when training metrics for tensorboard should be
  logged. Valid values: &#x27;epoch&#x27; and &#x27;batch&#x27;.
- `checkpoint_dir` - optional directory that should be used for model checkpointing
  

**Returns**:

  A list of callbacks.

#### update\_confidence\_type

```python
update_confidence_type(component_config: Dict[Text, Any]) -> Dict[Text, Any]
```

Set model confidence to auto if margin loss is used.

Option `auto` is reserved for margin loss type. It will be removed once margin loss
is deprecated.

**Arguments**:

- `component_config` - model configuration
  

**Returns**:

  updated model configuration

#### validate\_configuration\_settings

```python
validate_configuration_settings(component_config: Dict[Text, Any]) -> None
```

Validates that combination of parameters in the configuration are correctly set.

**Arguments**:

- `component_config` - Configuration to validate.

#### init\_split\_entities

```python
init_split_entities(split_entities_config: Union[bool, Dict[Text, Any]], default_split_entity: bool) -> Dict[Text, bool]
```

Initialise the behaviour for splitting entities by comma (or not).

**Returns**:

  Defines desired behaviour for splitting specific entity types and
  default behaviour for splitting any entity types for which no behaviour
  is defined.

