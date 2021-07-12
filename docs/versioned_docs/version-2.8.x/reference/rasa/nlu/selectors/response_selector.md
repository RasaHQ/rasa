---
sidebar_label: rasa.nlu.selectors.response_selector
title: rasa.nlu.selectors.response_selector
---
## ResponseSelector Objects

```python
class ResponseSelector(DIETClassifier)
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

#### \_\_init\_\_

```python
 | __init__(component_config: Optional[Dict[Text, Any]] = None, index_label_id_mapping: Optional[Dict[int, Text]] = None, entity_tag_specs: Optional[List[EntityTagSpec]] = None, model: Optional[RasaModel] = None, all_retrieval_intents: Optional[List[Text]] = None, responses: Optional[Dict[Text, List[Dict[Text, Any]]]] = None, finetune_mode: bool = False, sparse_feature_sizes: Optional[Dict[Text, Dict[Text, List[int]]]] = None) -> None
```

Declare instance variables with default values.

**Arguments**:

- `component_config` - Configuration for the component.
- `index_label_id_mapping` - Mapping between label and index used for encoding.
- `entity_tag_specs` - Format specification all entity tags.
- `model` - Model architecture.
- `all_retrieval_intents` - All retrieval intents defined in the data.
- `responses` - All responses defined in the data.
- `finetune_mode` - If `True` loads the model with pre-trained weights,
  otherwise initializes it with random weights.
- `sparse_feature_sizes` - Sizes of the sparse features the model was trained on.

#### preprocess\_train\_data

```python
 | preprocess_train_data(training_data: TrainingData) -> RasaModelData
```

Prepares data for training.

Performs sanity checks on training data, extracts encodings for labels.

**Arguments**:

- `training_data` - training data to preprocessed.

#### process

```python
 | process(message: Message, **kwargs: Any) -> None
```

Selects most like response for message.

**Arguments**:

- `message` - Latest user message.
- `kwargs` - Additional key word arguments.
  

**Returns**:

  the most likely response, the associated intent_response_key and its
  similarity to the input.

#### persist

```python
 | persist(file_name: Text, model_dir: Text) -> Dict[Text, Any]
```

Persist this model into the passed directory.

Return the metadata necessary to load the model again.

#### load

```python
 | @classmethod
 | load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Metadata = None, cached_component: Optional["ResponseSelector"] = None, **kwargs: Any, ,) -> "ResponseSelector"
```

Loads the trained model from the provided directory.

## DIET2DIET Objects

```python
class DIET2DIET(DIET)
```

Diet 2 Diet transformer implementation.

#### batch\_loss

```python
 | batch_loss(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> tf.Tensor
```

Calculates the loss for the given batch.

**Arguments**:

- `batch_in` - The batch.
  

**Returns**:

  The loss of the given batch.

#### batch\_predict

```python
 | batch_predict(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]
```

Predicts the output of the given batch.

**Arguments**:

- `batch_in` - The batch.
  

**Returns**:

  The output to predict.

