---
sidebar_label: rasa.nlu.classifiers._diet_classifier
title: rasa.nlu.classifiers._diet_classifier
---
## DIETClassifier Objects

```python
class DIETClassifier(IntentClassifier,  EntityExtractor)
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

#### \_\_init\_\_

```python
def __init__(component_config: Optional[Dict[Text, Any]] = None, index_label_id_mapping: Optional[Dict[int, Text]] = None, entity_tag_specs: Optional[List[EntityTagSpec]] = None, model: Optional[RasaModel] = None, finetune_mode: bool = False, sparse_feature_sizes: Optional[Dict[Text, Dict[Text, List[int]]]] = None) -> None
```

Declare instance variables with default values.

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
def train(training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any, ,) -> None
```

Train the embedding intent classifier on a data set.

#### process

```python
def process(message: Message, **kwargs: Any) -> None
```

Augments the message with intents, entities, and diagnostic data.

#### persist

```python
def persist(file_name: Text, model_dir: Text) -> Dict[Text, Any]
```

Persist this model into the passed directory.

Return the metadata necessary to load the model again.

#### load

```python
@classmethod
def load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Metadata = None, cached_component: Optional["DIETClassifier"] = None, should_finetune: bool = False, **kwargs: Any, ,) -> "DIETClassifier"
```

Loads the trained model from the provided directory.

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

