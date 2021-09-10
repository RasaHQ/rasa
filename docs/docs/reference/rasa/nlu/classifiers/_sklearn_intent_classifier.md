---
sidebar_label: rasa.nlu.classifiers._sklearn_intent_classifier
title: rasa.nlu.classifiers._sklearn_intent_classifier
---
## SklearnIntentClassifier Objects

```python
class SklearnIntentClassifier(IntentClassifier)
```

Intent classifier using the sklearn framework

#### \_\_init\_\_

```python
def __init__(component_config: Optional[Dict[Text, Any]] = None, clf: "sklearn.model_selection.GridSearchCV" = None, le: Optional["sklearn.preprocessing.LabelEncoder"] = None) -> None
```

Construct a new intent classifier using the sklearn framework.

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
def train(training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any, ,) -> None
```

Train the intent classifier on a data set.

#### process

```python
def process(message: Message, **kwargs: Any) -> None
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

vector of probabilities containing one entry for each label

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
def persist(file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]
```

Persist this model into the passed directory.

#### load

```python
@classmethod
def load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Optional[Metadata] = None, cached_component: Optional["SklearnIntentClassifier"] = None, **kwargs: Any, ,) -> "SklearnIntentClassifier"
```

Loads trained component (see parent class for full docstring).

