---
sidebar_label: rasa.nlu.classifiers.sklearn_intent_classifier
title: rasa.nlu.classifiers.sklearn_intent_classifier
---
## SklearnIntentClassifier Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class SklearnIntentClassifier(GraphComponent,  IntentClassifier)
```

Intent classifier using the sklearn framework.

#### required\_components

```python
 | @classmethod
 | required_components(cls) -> List[Type]
```

Components that should be included in the pipeline before this component.

#### get\_default\_config

```python
 | @staticmethod
 | get_default_config() -> Dict[Text, Any]
```

The component&#x27;s default config (see parent class for full docstring).

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, clf: "sklearn.model_selection.GridSearchCV" = None, le: Optional["sklearn.preprocessing.LabelEncoder"] = None) -> None
```

Construct a new intent classifier using the sklearn framework.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> SklearnIntentClassifier
```

Creates a new untrained component (see parent class for full docstring).

#### required\_packages

```python
 | @classmethod
 | required_packages(cls) -> List[Text]
```

Any extra python dependencies required for this component to run.

#### transform\_labels\_str2num

```python
 | transform_labels_str2num(labels: List[Text]) -> np.ndarray
```

Transforms a list of strings into numeric label representation.

**Arguments**:

- `labels`: List of labels to convert to numeric representation

#### transform\_labels\_num2str

```python
 | transform_labels_num2str(y: np.ndarray) -> np.ndarray
```

Transforms a list of strings into numeric label representation.

**Arguments**:

- `y`: List of labels to convert to numeric representation

#### train

```python
 | train(training_data: TrainingData) -> Resource
```

Train the intent classifier on a data set.

#### process

```python
 | process(messages: List[Message]) -> List[Message]
```

Return the most likely intent and its probability for a message.

#### predict\_prob

```python
 | predict_prob(X: np.ndarray) -> np.ndarray
```

Given a bow vector of an input text, predict the intent label.

Return probabilities for all labels.

**Arguments**:

- `X`: bow of input text

**Returns**:

vector of probabilities containing one entry for each label.

#### predict

```python
 | predict(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

Given a bow vector of an input text, predict most probable label.

Return only the most likely label.

**Arguments**:

- `X`: bow of input text

**Returns**:

tuple of first, the most probable label and second,

#### persist

```python
 | persist() -> None
```

Persist this model into the passed directory.

#### load

```python
 | @classmethod
 | load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> SklearnIntentClassifier
```

Loads trained component (see parent class for full docstring).

