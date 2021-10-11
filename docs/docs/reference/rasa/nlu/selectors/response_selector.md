---
sidebar_label: rasa.nlu.selectors.response_selector
title: rasa.nlu.selectors.response_selector
---
## ResponseSelectorGraphComponent Objects

```python
class ResponseSelectorGraphComponent(DIETClassifierGraphComponent)
```

Response selector using supervised embeddings.

The response selector embeds user inputs
and candidate response into the same space.
Supervised embeddings are trained by maximizing similarity between them.
It also provides rankings of the response that did not &quot;win&quot;.

The supervised response selector needs to be preceded by
a featurizer in the pipeline.
This featurizer creates the features used for the embeddings.
It is recommended to use ``CountVectorsFeaturizer`` that
can be optionally preceded by ``SpacyNLP`` and ``SpacyTokenizer``.

Based on the starspace idea from: https://arxiv.org/abs/1709.03856.
However, in this implementation the `mu` parameter is treated differently
and additional hidden layers are added together with dropout.

#### required\_components

```python
@classmethod
def required_components(cls) -> List[Type]
```

Components that should be included in the pipeline before this component.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

The component&#x27;s default config (see parent class for full docstring).

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, index_label_id_mapping: Optional[Dict[int, Text]] = None, entity_tag_specs: Optional[List[EntityTagSpec]] = None, model: Optional[RasaModel] = None, all_retrieval_intents: Optional[List[Text]] = None, responses: Optional[Dict[Text, List[Dict[Text, Any]]]] = None, sparse_feature_sizes: Optional[Dict[Text, Dict[Text, List[int]]]] = None) -> None
```

Declare instance variables with default values.

**Arguments**:

- `config` - Configuration for the component.
- `model_storage` - Storage which graph components can use to persist and load
  themselves.
- `resource` - Resource locator for this component which can be used to persist
  and load itself from the `model_storage`.
- `execution_context` - Information about the current graph run.
- `index_label_id_mapping` - Mapping between label and index used for encoding.
- `entity_tag_specs` - Format specification all entity tags.
- `model` - Model architecture.
- `all_retrieval_intents` - All retrieval intents defined in the data.
- `responses` - All responses defined in the data.
- `finetune_mode` - If `True` loads the model with pre-trained weights,
  otherwise initializes it with random weights.
- `sparse_feature_sizes` - Sizes of the sparse features the model was trained on.

#### label\_key

```python
@property
def label_key() -> Text
```

Returns label key.

#### label\_sub\_key

```python
@property
def label_sub_key() -> Text
```

Returns label sub_key.

#### model\_class

```python
@staticmethod
def model_class(use_text_as_label: bool) -> Type[RasaModel]
```

Returns model class.

#### preprocess\_train\_data

```python
def preprocess_train_data(training_data: TrainingData) -> RasaModelData
```

Prepares data for training.

Performs sanity checks on training data, extracts encodings for labels.

**Arguments**:

- `training_data` - training data to preprocessed.

#### process

```python
def process(messages: List[Message]) -> List[Message]
```

Selects most like response for message.

**Arguments**:

- `messages` - List containing latest user message.
  

**Returns**:

  List containing the message augmented with the most likely response,
  the associated intent_response_key and its similarity to the input.

#### persist

```python
def persist() -> None
```

Persist this model into the passed directory.

#### load

```python
@classmethod
def load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> ResponseSelectorGraphComponent
```

Loads the trained model from the provided directory.

## DIET2BOW Objects

```python
class DIET2BOW(DIET)
```

DIET2BOW transformer implementation.

## DIET2DIET Objects

```python
class DIET2DIET(DIET)
```

Diet 2 Diet transformer implementation.

#### batch\_loss

```python
def batch_loss(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> tf.Tensor
```

Calculates the loss for the given batch.

**Arguments**:

- `batch_in` - The batch.
  

**Returns**:

  The loss of the given batch.

#### batch\_predict

```python
def batch_predict(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]
```

Predicts the output of the given batch.

**Arguments**:

- `batch_in` - The batch.
  

**Returns**:

  The output to predict.

