---
sidebar_label: regex_featurizer
title: rasa.nlu.featurizers.sparse_featurizer.regex_featurizer
---

## RegexFeaturizer Objects

```python
class RegexFeaturizer(SparseFeaturizer)
```

#### \_\_init\_\_

```python
 | __init__(component_config: Optional[Dict[Text, Any]] = None, known_patterns: Optional[List[Dict[Text, Text]]] = None, pattern_vocabulary_stats: Optional[Dict[Text, int]] = None, finetune_mode: bool = False) -> None
```

Constructs new features for regexes and lookup table using regex expressions.

**Arguments**:

- `component_config` - Configuration for the component
- `known_patterns` - Regex Patterns the component should pre-load itself with.
- `pattern_vocabulary_stats` - Statistics about number of pattern slots filled and total number available.
- `finetune_mode` - Load component in finetune mode.

#### vocabulary\_stats

```python
 | @lazy_property
 | vocabulary_stats() -> Dict[Text, int]
```

Computes total vocabulary size and how much of it is consumed.

**Returns**:

  Computed vocabulary size and number of filled vocabulary slots.

#### train

```python
 | train(training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any, ,) -> None
```

Trains the component with all patterns extracted from training data.

**Arguments**:

- `training_data` - Training data consisting of training examples and patterns available.
- `config` - NLU Pipeline config
- `**kwargs` - Any other arguments

#### load

```python
 | @classmethod
 | load(cls, meta: Dict[Text, Any], model_dir: Optional[Text] = None, model_metadata: Optional[Metadata] = None, cached_component: Optional["RegexFeaturizer"] = None, should_finetune: bool = False, **kwargs: Any, ,) -> "RegexFeaturizer"
```

Loads a previously trained component.

**Arguments**:

- `meta` - Configuration of trained component.
- `model_dir` - Path where trained pipeline is stored.
- `model_metadata` - Metadata for the trained pipeline.
- `cached_component` - Previously cached component(if any).
- `should_finetune` - Indicates whether to load the component for further finetuning.
- `**kwargs` - Any other arguments.

#### persist

```python
 | persist(file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]
```

Persist this model into the passed directory.

**Arguments**:

- `file_name` - Prefix to add to all files stored as part of this component.
- `model_dir` - Path where files should be stored.
  

**Returns**:

  Metadata necessary to load the model again.

