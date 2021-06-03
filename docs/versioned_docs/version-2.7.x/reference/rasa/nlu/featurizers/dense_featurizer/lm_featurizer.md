---
sidebar_label: rasa.nlu.featurizers.dense_featurizer.lm_featurizer
title: rasa.nlu.featurizers.dense_featurizer.lm_featurizer
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
 | __init__(component_config: Optional[Dict[Text, Any]] = None, skip_model_load: bool = False, hf_transformers_loaded: bool = False) -> None
```

Initializes LanguageModelFeaturizer with the specified model.

**Arguments**:

- `component_config` - Configuration for the component.
- `skip_model_load` - Skip loading the model for pytests.
- `hf_transformers_loaded` - Skip loading of model and metadata, use
  HFTransformers output instead.

#### load

```python
 | @classmethod
 | load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Optional["Metadata"] = None, cached_component: Optional["Component"] = None, **kwargs: Any, ,) -> "Component"
```

Load this component from file.

After a component has been trained, it will be persisted by
calling `persist`. When the pipeline gets loaded again,
this component needs to be able to restore itself.
Components can rely on any context attributes that are
created by :meth:`components.Component.create`
calls to components previous to this one.

This method differs from the parent method only in that it calls create
rather than the constructor if the component is not found. This is to
trigger the check for HFTransformersNLP and the method can be removed
when HFTRansformersNLP is removed.

**Arguments**:

- `meta` - Any configuration parameter related to the model.
- `model_dir` - The directory to load the component from.
- `model_metadata` - The model&#x27;s :class:`rasa.nlu.model.Metadata`.
- `cached_component` - The cached component.
  

**Returns**:

  the loaded component

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

