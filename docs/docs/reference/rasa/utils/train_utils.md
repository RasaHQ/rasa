---
sidebar_label: rasa.utils.train_utils
title: rasa.utils.train_utils
---
#### normalize

```python
def normalize(values: np.ndarray, ranking_length: int = 0) -> np.ndarray
```

Normalizes an array of positive numbers over the top `ranking_length` values.

Other values will be set to 0.

#### update\_similarity\_type

```python
def update_similarity_type(config: Dict[Text, Any]) -> Dict[Text, Any]
```

If SIMILARITY_TYPE is set to &#x27;auto&#x27;, update the SIMILARITY_TYPE depending
on the LOSS_TYPE.

**Arguments**:

- `config` - model configuration
  
- `Returns` - updated model configuration

#### align\_token\_features

```python
def align_token_features(list_of_tokens: List[List["Token"]], in_token_features: np.ndarray, shape: Optional[Tuple] = None) -> np.ndarray
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
def update_evaluation_parameters(config: Dict[Text, Any]) -> Dict[Text, Any]
```

If EVAL_NUM_EPOCHS is set to -1, evaluate at the end of the training.

**Arguments**:

- `config` - model configuration
  
- `Returns` - updated model configuration

#### load\_tf\_hub\_model

```python
def load_tf_hub_model(model_url: Text) -> Any
```

Load model from cache if possible, otherwise from TFHub

#### check\_deprecated\_options

```python
def check_deprecated_options(config: Dict[Text, Any]) -> Dict[Text, Any]
```

Update the config according to changed config params.

If old model configuration parameters are present in the provided config, replace
them with the new parameters and log a warning.

**Arguments**:

- `config` - model configuration
  
- `Returns` - updated model configuration

#### check\_core\_deprecated\_options

```python
def check_core_deprecated_options(config: Dict[Text, Any]) -> Dict[Text, Any]
```

Update the core config according to changed config params.

If old model configuration parameters are present in the provided config, replace
them with the new parameters and log a warning.

**Arguments**:

- `config` - model configuration
  
- `Returns` - updated model configuration

#### entity\_label\_to\_tags

```python
def entity_label_to_tags(model_predictions: Dict[Text, Any], entity_tag_specs: List["EntityTagSpec"], bilou_flag: bool = False, prediction_index: int = 0) -> Tuple[Dict[Text, List[Text]], Dict[Text, List[float]]]
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

#### override\_defaults

```python
def override_defaults(defaults: Optional[Dict[Text, Any]], custom: Optional[Dict[Text, Any]]) -> Dict[Text, Any]
```

Override default config with the given config.

We cannot use `dict.update` method because configs contain nested dicts.

**Arguments**:

- `defaults` - default config
- `custom` - user config containing new parameters
  

**Returns**:

  updated config

#### create\_data\_generators

```python
def create_data_generators(model_data: RasaModelData, batch_sizes: Union[int, List[int]], epochs: int, batch_strategy: Text = SEQUENCE, eval_num_examples: int = 0, random_seed: Optional[int] = None, shuffle: bool = True) -> Tuple[RasaBatchDataGenerator, Optional[RasaBatchDataGenerator]]
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
def create_common_callbacks(epochs: int, tensorboard_log_dir: Optional[Text] = None, tensorboard_log_level: Optional[Text] = None, checkpoint_dir: Optional[Path] = None) -> List["Callback"]
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
def update_confidence_type(component_config: Dict[Text, Any]) -> Dict[Text, Any]
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
def validate_configuration_settings(component_config: Dict[Text, Any]) -> None
```

Validates that combination of parameters in the configuration are correctly set.

**Arguments**:

- `component_config` - Configuration to validate.

#### init\_split\_entities

```python
def init_split_entities(split_entities_config: Union[bool, Dict[Text, Any]], default_split_entity: bool) -> Dict[Text, bool]
```

Initialise the behaviour for splitting entities by comma (or not).

**Returns**:

  Defines desired behaviour for splitting specific entity types and
  default behaviour for splitting any entity types for which no behaviour
  is defined.

