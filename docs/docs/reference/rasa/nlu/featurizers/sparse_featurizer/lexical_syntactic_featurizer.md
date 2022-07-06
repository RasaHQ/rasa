---
sidebar_label: rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer
title: rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer
---
## LexicalSyntacticFeaturizer Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=True
)
class LexicalSyntacticFeaturizer(SparseFeaturizer,  GraphComponent)
```

Extracts and encodes lexical syntactic features.

Given a sequence of tokens, this featurizer produces a sequence of features
where the `t`-th feature encodes lexical and syntactic information about the `t`-th
token and it&#x27;s surrounding tokens.

In detail: The lexical syntactic features can be specified via a list of
configurations `[c_0, c_1, ..., c_n]` where each `c_i` is a list of names of
lexical and syntactic features (e.g. `low`, `suffix2`, `digit`).
For a given tokenized text, the featurizer will consider a window of size `n`
around each token and evaluate the given list of configurations as follows:
- It will extract the features listed in `c_m` where `m = (n-1)/2` if n is even and
`n/2` from token `t`
- It will extract the features listed in `c_{m-1}`,`c_{m-2}` ... ,  from the last,
second to last, ... token before token `t`, respectively.
- It will extract the features listed `c_{m+1}`, `c_{m+1}`, ... for the first,
second, ... token `t`, respectively.
It will then combine all these features into one feature for position `t`.

**Example**:

  If we specify `[[&#x27;low&#x27;], [&#x27;upper&#x27;], [&#x27;prefix2&#x27;]]`, then for each position `t`
  the `t`-th feature will encode whether the token at position `t` is upper case,
  where the token at position `t-1` is lower case and the first two characters
  of the token at position `t+1`.

#### required\_components

```python
@classmethod
def required_components(cls) -> List[Type]
```

Components that should be included in the pipeline before this component.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

Returns the component&#x27;s default config.

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, feature_to_idx_dict: Optional[Dict[Tuple[int, Text], Dict[Text, int]]] = None) -> None
```

Instantiates a new `LexicalSyntacticFeaturizer` instance.

#### validate\_config

```python
@classmethod
def validate_config(cls, config: Dict[Text, Any]) -> None
```

Validates that the component is configured properly.

#### train

```python
def train(training_data: TrainingData) -> Resource
```

Trains the featurizer.

**Arguments**:

- `training_data` - the training data
  

**Returns**:

  the resource from which this trained component can be loaded

#### warn\_if\_pos\_features\_cannot\_be\_computed

```python
def warn_if_pos_features_cannot_be_computed(training_data: TrainingData) -> None
```

Warn if part-of-speech features are needed but not given.

#### process

```python
def process(messages: List[Message]) -> List[Message]
```

Featurizes all given messages in-place.

**Arguments**:

- `messages` - messages to be featurized.
  

**Returns**:

  The same list with the same messages after featurization.

#### process\_training\_data

```python
def process_training_data(training_data: TrainingData) -> TrainingData
```

Processes the training examples in the given training data in-place.

**Arguments**:

- `training_data` - the training data
  

**Returns**:

  same training data after processing

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> LexicalSyntacticFeaturizer
```

Creates a new untrained component (see parent class for full docstring).

#### load

```python
@classmethod
def load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> LexicalSyntacticFeaturizer
```

Loads trained component (see parent class for full docstring).

#### persist

```python
def persist() -> None
```

Persist this model (see parent class for full docstring).

