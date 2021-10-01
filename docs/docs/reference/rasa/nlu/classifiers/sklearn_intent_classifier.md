---
sidebar_label: rasa.nlu.classifiers.sklearn_intent_classifier
title: rasa.nlu.classifiers.sklearn_intent_classifier
---
## SklearnIntentClassifierGraphComponent Objects

```python
class SklearnIntentClassifierGraphComponent(GraphComponent)
```

Intent classifier using the sklearn framework.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

The component&#x27;s default config (see parent class for full docstring).

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, clf: "sklearn.model_selection.GridSearchCV" = None, le: Optional["sklearn.preprocessing.LabelEncoder"] = None) -> None
```

Construct a new intent classifier using the sklearn framework.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> SklearnIntentClassifierGraphComponent
```

Creates a new untrained component (see parent class for full docstring).

#### required\_packages

```python
@classmethod
def required_packages(cls) -> List[Text]
```

Any extra python dependencies required for this component to run.

#### transform\_labels\_str2num

```python
def transform_labels_str2num(labels: List[Text]) -> np.ndarray
```

Transforms a list of strings into numeric label representation.

**Arguments**:

- `labels`: List of labels to convert to numeric representation

#### transform\_labels\_num2str

```python
def transform_labels_num2str(y: np.ndarray) -> np.ndarray
```

Transforms a list of strings into numeric label representation.

**Arguments**:

- `y`: List of labels to convert to numeric representation

#### train

```python
def train(training_data: TrainingData) -> Resource
```

Train the intent classifier on a data set.

#### process

```python
def process(messages: List[Message]) -> List[Message]
```

Return the most likely intent and its probability for a message.

#### predict\_prob

```python
def predict_prob(X: np.ndarray) -> np.ndarray
```

Given a bow vector of an input text, predict the intent label.

Return probabilities for all labels.

**Arguments**:

- `X`: bow of input text

**Returns**:

vector of probabilities containing one entry for each label.

#### predict

```python
def predict(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

Given a bow vector of an input text, predict most probable label.

Return only the most likely label.

**Arguments**:

- `X`: bow of input text

**Returns**:

tuple of first, the most probable label and second,

#### persist

```python
def persist() -> None
```

Persist this model into the passed directory.

#### load

```python
@classmethod
def load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> SklearnIntentClassifierGraphComponent
```

Loads trained component (see parent class for full docstring).

