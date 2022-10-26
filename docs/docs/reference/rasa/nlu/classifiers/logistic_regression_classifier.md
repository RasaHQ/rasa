---
sidebar_label: rasa.nlu.classifiers.logistic_regression_classifier
title: rasa.nlu.classifiers.logistic_regression_classifier
---
## LogisticRegressionClassifier Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class LogisticRegressionClassifier(IntentClassifier, GraphComponent)
```

Intent classifier using the Logistic Regression.

#### required\_components

```python
@classmethod
def required_components(cls) -> List[Type]
```

Components that should be included in the pipeline before this component.

#### required\_packages

```python
@staticmethod
def required_packages() -> List[Text]
```

Any extra python dependencies required for this component to run.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

The component&#x27;s default config (see parent class for full docstring).

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any], name: Text, model_storage: ModelStorage,
             resource: Resource) -> None
```

Construct a new classifier.

#### train

```python
def train(training_data: TrainingData) -> Resource
```

Train the intent classifier on a data set.

#### create

```python
@classmethod
def create(
        cls, config: Dict[Text, Any], model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext) -> "LogisticRegressionClassifier"
```

Creates a new untrained component (see parent class for full docstring).

#### process

```python
def process(messages: List[Message]) -> List[Message]
```

Return the most likely intent and its probability for a message.

#### persist

```python
def persist() -> None
```

Persist this model into the passed directory.

#### load

```python
@classmethod
def load(cls, config: Dict[Text, Any], model_storage: ModelStorage,
         resource: Resource, execution_context: ExecutionContext,
         **kwargs: Any) -> "LogisticRegressionClassifier"
```

Loads trained component (see parent class for full docstring).

#### process\_training\_data

```python
def process_training_data(training_data: TrainingData) -> TrainingData
```

Process the training data.

#### validate\_config

```python
@classmethod
def validate_config(cls, config: Dict[Text, Any]) -> None
```

Validates that the component is configured properly.

