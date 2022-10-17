---
sidebar_label: rasa.nlu.featurizers.dense_featurizer._lm_featurizer
title: rasa.nlu.featurizers.dense_featurizer._lm_featurizer
---
## LanguageModelFeaturizer Objects

```python
class LanguageModelFeaturizer(DenseFeaturizer)
```

Featurizer using transformer-based language models.

The transformers(https://github.com/huggingface/transformers) library
is used to load pre-trained language models like BERT, GPT-2, etc.
The component also tokenizes and featurizes dense featurizable attributes of
each message.

#### required\_components

```python
 | @classmethod
 | required_components(cls) -> List[Type[Component]]
```

Packages needed to be installed.

#### \_\_init\_\_

```python
 | __init__(component_config: Optional[Dict[Text, Any]] = None, skip_model_load: bool = False) -> None
```

Initializes LanguageModelFeaturizer with the specified model.

**Arguments**:

- `component_config` - Configuration for the component.
- `skip_model_load` - Skip loading the model for pytests.

#### cache\_key

```python
 | @classmethod
 | cache_key(cls, component_meta: Dict[Text, Any], model_metadata: Metadata) -> Optional[Text]
```

Cache the component for future use.

**Arguments**:

- `component_meta` - configuration for the component.
- `model_metadata` - configuration for the whole pipeline.
  
- `Returns` - key of the cache for future retrievals.

#### required\_packages

```python
 | @classmethod
 | required_packages(cls) -> List[Text]
```

Packages needed to be installed.

#### train

```python
 | train(training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any, ,) -> None
```

Compute tokens and dense features for each message in training data.

**Arguments**:

- `training_data` - NLU training data to be tokenized and featurized
- `config` - NLU pipeline config consisting of all components.

#### process

```python
 | process(message: Message, **kwargs: Any) -> None
```

Process an incoming message by computing its tokens and dense features.

**Arguments**:

- `message` - Incoming message object

