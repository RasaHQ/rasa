---
sidebar_label: rasa.nlu.featurizers.dense_featurizer.lm_featurizer
title: rasa.nlu.featurizers.dense_featurizer.lm_featurizer
---
## LanguageModelFeaturizerGraphComponent Objects

```python
class LanguageModelFeaturizerGraphComponent(DenseFeaturizer2,  GraphComponent)
```

A featurizer that uses transformer-based language models.

This component loads a pre-trained language model
from the Transformers library (https://github.com/huggingface/transformers)
including BERT, GPT, GPT-2, xlnet, distilbert, and roberta.
It also tokenizes and featurizes the featurizable dense attributes of
each message.

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

#### validate\_compatibility\_with\_tokenizer

```python
@classmethod
def validate_compatibility_with_tokenizer(cls, config: Dict[Text, Any], tokenizer_type: Type[Tokenizer]) -> None
```

Checks that the featurizer and tokenizer are compatible.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> LanguageModelFeaturizerGraphComponent
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
def process_training_data(training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any, ,) -> TrainingData
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

