---
sidebar_label: rasa.nlu.featurizers.sparse_featurizer._regex_featurizer
title: rasa.nlu.featurizers.sparse_featurizer._regex_featurizer
---
## RegexFeaturizer Objects

```python
class RegexFeaturizer(SparseFeaturizer)
```

#### \_\_init\_\_

```python
def __init__(component_config: Optional[Dict[Text, Any]] = None, known_patterns: Optional[List[Dict[Text, Text]]] = None, finetune_mode: bool = False) -> None
```

Constructs new features for regexes and lookup table using regex expressions.

**Arguments**:

- `component_config` - Configuration for the component
- `known_patterns` - Regex Patterns the component should pre-load itself with.
- `finetune_mode` - Load component in finetune mode.

#### train

```python
def train(training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any, ,) -> None
```

Trains the component with all patterns extracted from training data.

**Arguments**:

- `training_data` - Training data consisting of training examples and patterns
  available.
- `config` - NLU Pipeline config
- `**kwargs` - Any other arguments

#### load

```python
@classmethod
def load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Optional[Metadata] = None, cached_component: Optional["RegexFeaturizer"] = None, should_finetune: bool = False, **kwargs: Any, ,) -> "RegexFeaturizer"
```

Loads a previously trained component.

**Arguments**:

- `meta` - Configuration of trained component.
- `model_dir` - Path where trained pipeline is stored.
- `model_metadata` - Metadata for the trained pipeline.
- `cached_component` - Previously cached component(if any).
- `should_finetune` - Indicates whether to load the component for further
  finetuning.
- `**kwargs` - Any other arguments.

#### persist

```python
def persist(file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]
```

Persist this model into the passed directory.

**Arguments**:

- `file_name` - Prefix to add to all files stored as part of this component.
- `model_dir` - Path where files should be stored.
  

**Returns**:

  Metadata necessary to load the model again.

