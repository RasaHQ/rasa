---
sidebar_label: rasa.nlu.featurizers.sparse_featurizer._count_vectors_featurizer
title: rasa.nlu.featurizers.sparse_featurizer._count_vectors_featurizer
---
## CountVectorsFeaturizer Objects

```python
class CountVectorsFeaturizer(SparseFeaturizer)
```

Creates a sequence of token counts features based on sklearn&#x27;s `CountVectorizer`.

All tokens which consist only of digits (e.g. 123 and 99
but not ab12d) will be represented by a single feature.

Set `analyzer` to &#x27;char_wb&#x27;
to use the idea of Subword Semantic Hashing
from https://arxiv.org/abs/1810.07150.

#### \_\_init\_\_

```python
def __init__(component_config: Optional[Dict[Text, Any]] = None, vectorizers: Optional[Dict[Text, "CountVectorizer"]] = None, finetune_mode: bool = False) -> None
```

Construct a new count vectorizer using the sklearn framework.

#### train

```python
def train(training_data: TrainingData, cfg: Optional[RasaNLUModelConfig] = None, **kwargs: Any, ,) -> None
```

Train the featurizer.

Take parameters from config and
construct a new count vectorizer using the sklearn framework.

#### process

```python
def process(message: Message, **kwargs: Any) -> None
```

Process incoming message and compute and set features

#### persist

```python
def persist(file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]
```

Persist this model into the passed directory.

Returns the metadata necessary to load the model again.

#### load

```python
@classmethod
def load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Optional[Metadata] = None, cached_component: Optional["CountVectorsFeaturizer"] = None, should_finetune: bool = False, **kwargs: Any, ,) -> "CountVectorsFeaturizer"
```

Loads trained component (see parent class for full docstring).

