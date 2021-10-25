---
sidebar_label: rasa.nlu.featurizers.dense_featurizer.lm_featurizer
title: rasa.nlu.featurizers.dense_featurizer.lm_featurizer
---
## LanguageModelFeaturizer Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class LanguageModelFeaturizer(DenseFeaturizer,  GraphComponent)
```

A featurizer that uses transformer-based language models.

This component loads a pre-trained language model
from the Transformers library (https://github.com/huggingface/transformers)
including BERT, GPT, GPT-2, xlnet, distilbert, and roberta.
It also tokenizes and featurizes the featurizable dense attributes of
each message.

#### required\_components

```python
@classmethod
def required_components(cls) -> List[Type]
```

Components that should be included in the pipeline before this component.

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any], execution_context: ExecutionContext) -> None
```

Initializes the featurizer with the model in the config.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

Returns LanguageModelFeaturizer&#x27;s default config.

#### validate\_config

```python
@classmethod
def validate_config(cls, config: Dict[Text, Any]) -> None
```

Validates the configuration.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> LanguageModelFeaturizer
```

Creates a LanguageModelFeaturizer.

Loads the model specified in the config.

#### required\_packages

```python
@staticmethod
def required_packages() -> List[Text]
```

Returns the extra python dependencies required.

#### process\_training\_data

```python
def process_training_data(training_data: TrainingData) -> TrainingData
```

Computes tokens and dense features for each message in training data.

**Arguments**:

- `training_data` - NLU training data to be tokenized and featurized
- `config` - NLU pipeline config consisting of all components.

#### process

```python
def process(messages: List[Message]) -> List[Message]
```

Processes messages by computing tokens and dense features.

