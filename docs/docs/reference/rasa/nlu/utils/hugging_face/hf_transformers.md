---
sidebar_label: rasa.nlu.utils.hugging_face.hf_transformers
title: rasa.nlu.utils.hugging_face.hf_transformers
---
## HFTransformersNLP Objects

```python
class HFTransformersNLP(Component)
```

This component is deprecated and will be removed in the future.

Use the LanguageModelFeaturizer instead.

#### \_\_init\_\_

```python
 | __init__(component_config: Optional[Dict[Text, Any]] = None, skip_model_load: bool = False) -> None
```

Initializes HFTransformsNLP with the models specified.

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

