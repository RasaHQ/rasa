---
sidebar_label: rasa.core.policies._ted_policy
title: rasa.core.policies._ted_policy
---
## TEDPolicy Objects

```python
class TEDPolicy(Policy)
```

Transformer Embedding Dialogue (TED) Policy.

The model architecture is described in
detail in https://arxiv.org/abs/1910.00486.
In summary, the architecture comprises of the
following steps:
    - concatenate user input (user intent and entities), previous system actions,
      slots and active forms for each time step into an input vector to
      pre-transformer embedding layer;
    - feed it to transformer;
    - apply a dense layer to the output of the transformer to get embeddings of a
      dialogue for each time step;
    - apply a dense layer to create embeddings for system actions for each time
      step;
    - calculate the similarity between the dialogue embedding and embedded system
      actions. This step is based on the StarSpace
      (https://arxiv.org/abs/1709.03856) idea.

#### \_\_init\_\_

```python
def __init__(featurizer: Optional[TrackerFeaturizer] = None, priority: int = DEFAULT_POLICY_PRIORITY, max_history: Optional[int] = None, model: Optional[RasaModel] = None, fake_features: Optional[Dict[Text, List["Features"]]] = None, entity_tag_specs: Optional[List[EntityTagSpec]] = None, should_finetune: bool = False, **kwargs: Any, ,) -> None
```

Declares instance variables with default values.

#### model\_class

```python
@staticmethod
def model_class() -> Type["TED"]
```

Gets the class of the model architecture to be used by the policy.

**Returns**:

  Required class.

#### run\_training

```python
def run_training(model_data: RasaModelData, label_ids: Optional[np.ndarray] = None) -> None
```

Feeds the featurized training data to the model.

**Arguments**:

- `model_data` - Featurized training data.
- `label_ids` - Label ids corresponding to the data points in `model_data`.
  These may or may not be used by the function depending
  on how the policy is trained.

#### train

```python
def train(training_trackers: List[TrackerWithCachedStates], domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> None
```

Trains the policy on given training trackers.

**Arguments**:

- `training_trackers` - List of training trackers to be used
  for training the model.
- `domain` - Domain of the assistant.
- `interpreter` - NLU Interpreter to be used for featurizing the states.
- `**kwargs` - Any other argument.

#### predict\_action\_probabilities

```python
def predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> PolicyPrediction
```

Predicts the next action the bot should take after seeing the tracker.

**Arguments**:

- `tracker` - the :class:`rasa.core.trackers.DialogueStateTracker`
- `domain` - the :class:`rasa.shared.core.domain.Domain`
- `interpreter` - Interpreter which may be used by the policies to create
  additional features.
  

**Returns**:

  The policy&#x27;s prediction (e.g. the probabilities for the actions).

#### persist

```python
def persist(path: Union[Text, Path]) -> None
```

Persists the policy to a storage.

#### persist\_model\_utilities

```python
def persist_model_utilities(model_path: Path) -> None
```

Persists model&#x27;s utility attributes like model weights, etc.

**Arguments**:

- `model_path` - Path where model is to be persisted

#### load

```python
@classmethod
def load(cls, path: Union[Text, Path], should_finetune: bool = False, epoch_override: int = defaults[EPOCHS], **kwargs: Any, ,) -> "TEDPolicy"
```

Loads a policy from the storage.

**Arguments**:

- `path` - Path on disk where policy is persisted.
- `should_finetune` - Whether to load the policy for finetuning.
- `epoch_override` - Override the number of epochs in persisted
  configuration for further finetuning.
- `**kwargs` - Any other arguments
  

**Returns**:

  Loaded policy
  

**Raises**:

  `PolicyModelNotFound` if the model is not found in the supplied `path`.

