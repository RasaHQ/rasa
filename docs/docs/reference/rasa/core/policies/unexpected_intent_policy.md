---
sidebar_label: rasa.core.policies.unexpected_intent_policy
title: rasa.core.policies.unexpected_intent_policy
---
## UnexpecTEDIntentPolicy Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITH_END_TO_END_SUPPORT, is_trainable=True
)
class UnexpecTEDIntentPolicy(TEDPolicy)
```

`UnexpecTEDIntentPolicy` has the same model architecture as `TEDPolicy`.

The difference is at a task level.
Instead of predicting the next probable action, this policy
predicts whether the last predicted intent is a likely intent
according to the training stories and conversation context.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

Returns the default config (see parent class for full docstring).

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, model: Optional[RasaModel] = None, featurizer: Optional[TrackerFeaturizer] = None, fake_features: Optional[Dict[Text, List[Features]]] = None, entity_tag_specs: Optional[List[EntityTagSpec]] = None, label_quantiles: Optional[Dict[int, List[float]]] = None)
```

Declares instance variables with default values.

#### model\_class

```python
@staticmethod
def model_class() -> Type["IntentTED"]
```

Gets the class of the model architecture to be used by the policy.

**Returns**:

  Required class.

#### compute\_label\_quantiles\_post\_training

```python
def compute_label_quantiles_post_training(model_data: RasaModelData, label_ids: np.ndarray) -> None
```

Computes quantile scores for prediction of `action_unlikely_intent`.

Multiple quantiles are computed for each label
so that an appropriate threshold can be picked at
inference time according to the `tolerance` value specified.

**Arguments**:

- `model_data` - Data used for training the model.
- `label_ids` - Numerical IDs of labels for each data point used during training.

#### run\_training

```python
def run_training(model_data: RasaModelData, label_ids: Optional[np.ndarray] = None) -> None
```

Feeds the featurized training data to the model.

**Arguments**:

- `model_data` - Featurized training data.
- `label_ids` - Label ids corresponding to the data points in `model_data`.
  

**Raises**:

  `RasaCoreException` if `label_ids` is None as it&#x27;s needed for
  running post training procedures.

#### predict\_action\_probabilities

```python
def predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, precomputations: Optional[MessageContainerForCoreFeaturization] = None, rule_only_data: Optional[Dict[Text, Any]] = None, **kwargs: Any, ,) -> PolicyPrediction
```

Predicts the next action the bot should take after seeing the tracker.

**Arguments**:

- `tracker` - Tracker containing past conversation events.
- `domain` - Domain of the assistant.
- `precomputations` - Contains precomputed features and attributes.
- `rule_only_data` - Slots and loops which are specific to rules and hence
  should be ignored by this policy.
  

**Returns**:

  The policy&#x27;s prediction (e.g. the probabilities for the actions).

#### persist\_model\_utilities

```python
def persist_model_utilities(model_path: Path) -> None
```

Persists model&#x27;s utility attributes like model weights, etc.

**Arguments**:

- `model_path` - Path where model is to be persisted

## IntentTED Objects

```python
class IntentTED(TED)
```

Follows TED&#x27;s model architecture from https://arxiv.org/abs/1910.00486.

However, it has been re-purposed to predict multiple
labels (intents) instead of a single label (action).

#### dot\_product\_loss\_layer

```python
@property
def dot_product_loss_layer() -> tf.keras.layers.Layer
```

Returns the dot-product loss layer to use.

Multiple intents can be valid simultaneously, so `IntentTED` uses the
`MultiLabelDotProductLoss`.

**Returns**:

  The loss layer that is used by `_prepare_dot_product_loss`.

#### run\_bulk\_inference

```python
def run_bulk_inference(model_data: RasaModelData) -> Dict[Text, np.ndarray]
```

Computes model&#x27;s predictions for input data.

**Arguments**:

- `model_data` - Data to be passed as input
  

**Returns**:

  Predictions for the input data.

