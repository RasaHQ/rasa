---
sidebar_label: rasa.core.policies.ted_policy
title: rasa.core.policies.ted_policy
---
## TEDPolicyGraphComponent Objects

```python
class TEDPolicyGraphComponent(PolicyGraphComponent)
```

Transformer Embedding Dialogue (TED) Policy.

The model architecture is described in
detail in https://arxiv.org/abs/1910.00486.
In summary, the architecture comprises of the
following steps:
    - concatenate user input (user intent and entities), previous system actions,
      slots and active forms for each time step into an input vector to
      pre-transformer embedding layer;
    - feed it to transformer;
    - apply a dense layer to the output of the transformer to get embeddings of a
      dialogue for each time step;
    - apply a dense layer to create embeddings for system actions for each time
      step;
    - calculate the similarity between the dialogue embedding and embedded system
      actions. This step is based on the StarSpace
      (https://arxiv.org/abs/1709.03856) idea.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

Returns the default config (see parent class for full docstring).

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, model: Optional[RasaModel] = None, featurizer: Optional[TrackerFeaturizer] = None, fake_features: Optional[Dict[Text, List[Features]]] = None, entity_tag_specs: Optional[List[EntityTagSpec]] = None) -> None
```

Declares instance variables with default values.

#### model\_class

```python
@staticmethod
def model_class() -> Type[TED]
```

Gets the class of the model architecture to be used by the policy.

**Returns**:

  Required class.

#### run\_training

```python
def run_training(model_data: RasaModelData, label_ids: Optional[np.ndarray] = None) -> None
```

Feeds the featurized training data to the model.

**Arguments**:

- `model_data` - Featurized training data.
- `label_ids` - Label ids corresponding to the data points in `model_data`.
  These may or may not be used by the function depending
  on how the policy is trained.

#### train

```python
def train(training_trackers: List[TrackerWithCachedStates], domain: Domain, precomputations: Optional[MessageContainerForCoreFeaturization] = None) -> Resource
```

Trains the policy (see parent class for full docstring).

#### predict\_action\_probabilities

```python
def predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, precomputations: Optional[MessageContainerForCoreFeaturization] = None, rule_only_data: Optional[Dict[Text, Any]] = None, **kwargs: Any, ,) -> PolicyPrediction
```

Predicts the next action (see parent class for full docstring).

#### persist

```python
def persist() -> None
```

Persists the policy to a storage.

#### persist\_model\_utilities

```python
def persist_model_utilities(model_path: Path) -> None
```

Persists model&#x27;s utility attributes like model weights, etc.

**Arguments**:

- `model_path` - Path where model is to be persisted

#### load

```python
@classmethod
def load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> TEDPolicyGraphComponent
```

Loads a policy from the storage (see parent class for full docstring).

## TED Objects

```python
class TED(TransformerRasaModel)
```

TED model architecture from https://arxiv.org/abs/1910.00486.

#### \_\_init\_\_

```python
def __init__(data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]], config: Dict[Text, Any], max_history_featurizer_is_used: bool, label_data: RasaModelData, entity_tag_specs: Optional[List[EntityTagSpec]]) -> None
```

Initializes the TED model.

**Arguments**:

- `data_signature` - the data signature of the input data
- `config` - the model configuration
- `max_history_featurizer_is_used` - if &#x27;True&#x27;
  only the last dialogue turn will be used
- `label_data` - the label data
- `entity_tag_specs` - the entity tag specifications

#### batch\_loss

```python
def batch_loss(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> tf.Tensor
```

Calculates the loss for the given batch.

**Arguments**:

- `batch_in` - The batch.
  

**Returns**:

  The loss of the given batch.

#### prepare\_for\_predict

```python
def prepare_for_predict() -> None
```

Prepares the model for prediction.

#### batch\_predict

```python
def batch_predict(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]
```

Predicts the output of the given batch.

**Arguments**:

- `batch_in` - The batch.
  

**Returns**:

  The output to predict.

