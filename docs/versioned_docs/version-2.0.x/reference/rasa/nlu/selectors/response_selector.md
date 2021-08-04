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

Return the most likely response, the associated intent_response_key and its similarity to the input.

#### persist

```python
 | persist(file_name: Text, model_dir: Text) -> Dict[Text, Any]
```

Persist this model into the passed directory.

Return the metadata necessary to load the model again.

#### load

```python
 | @classmethod
 | load(cls, meta: Dict[Text, Any], model_dir: Text = None, model_metadata: Metadata = None, cached_component: Optional["ResponseSelector"] = None, **kwargs: Any, ,) -> "ResponseSelector"
```

Loads the trained model from the provided directory.

