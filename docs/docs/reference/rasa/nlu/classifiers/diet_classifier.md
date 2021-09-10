---
sidebar_label: rasa.nlu.classifiers.diet_classifier
title: rasa.nlu.classifiers.diet_classifier
---
## DIETClassifierGraphComponent Objects

```python
class DIETClassifierGraphComponent(GraphComponent,  EntityExtractorMixin)
```

A multi-task model for intent classification and entity extraction.

DIET is Dual Intent and Entity Transformer.
The architecture is based on a transformer which is shared for both tasks.
A sequence of entity labels is predicted through a Conditional Random Field (CRF)
tagging layer on top of the transformer output sequence corresponding to the
input sequence of tokens. The transformer output for the ``__CLS__`` token and
intent labels are embedded into a single semantic vector space. We use the
dot-product loss to maximize the similarity with the target label and minimize
similarities with negative samples.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

The component&#x27;s default config (see parent class for full docstring).

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, index_label_id_mapping: Optional[Dict[int, Text]] = None, entity_tag_specs: Optional[List[EntityTagSpec]] = None, model: Optional[RasaModel] = None, sparse_feature_sizes: Optional[Dict[Text, Dict[Text, List[int]]]] = None) -> None
```

Declare instance variables with default values.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> DIETClassifierGraphComponent
```

Creates a new untrained component (see parent class for full docstring).

#### label\_key

```python
@property
def label_key() -> Optional[Text]
```

Return key if intent classification is activated.

#### label\_sub\_key

```python
@property
def label_sub_key() -> Optional[Text]
```

Return sub key if intent classification is activated.

#### preprocess\_train\_data

```python
def preprocess_train_data(training_data: TrainingData) -> RasaModelData
```

Prepares data for training.

Performs sanity checks on training data, extracts encodings for labels.

#### train

```python
def train(training_data: TrainingData) -> Resource
```

Train the embedding intent classifier on a data set.

#### process

```python
def process(messages: List[Message]) -> List[Message]
```

Augments the message with intents, entities, and diagnostic data.

#### persist

```python
def persist() -> None
```

Persist this model into the passed directory.

#### load

```python
@classmethod
def load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> DIETClassifierGraphComponent
```

Loads a policy from the storage (see parent class for full docstring).

## DIET Objects

```python
class DIET(TransformerRasaModel)
```

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
def batch_predict(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> Dict[Text, tf.Tensor]
```

Predicts the output of the given batch.

**Arguments**:

- `batch_in` - The batch.
  

**Returns**:

  The output to predict.

