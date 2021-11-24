---
sidebar_label: rasa.nlu.tokenizers.jieba_tokenizer
title: rasa.nlu.tokenizers.jieba_tokenizer
---
## JiebaTokenizer Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=True
)
class JiebaTokenizer(Tokenizer)
```

This tokenizer is a wrapper for Jieba (https://github.com/fxsjy/jieba).

#### supported\_languages

```python
 | @staticmethod
 | supported_languages() -> Optional[List[Text]]
```

Supported languages (see parent class for full docstring).

#### get\_default\_config

```python
 | @staticmethod
 | get_default_config() -> Dict[Text, Any]
```

Returns default config (see parent class for full docstring).

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource) -> None
```

Initialize the tokenizer.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> JiebaTokenizer
```

Creates a new component (see parent class for full docstring).

#### required\_packages

```python
 | @classmethod
 | required_packages(cls) -> List[Text]
```

Any extra python dependencies required for this component to run.

#### train

```python
 | train(training_data: TrainingData) -> Resource
```

Copies the dictionary to the model storage.

#### tokenize

```python
 | tokenize(message: Message, attribute: Text) -> List[Token]
```

Tokenizes the text of the provided attribute of the incoming message.

#### load

```python
 | @classmethod
 | load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> JiebaTokenizer
```

Loads a custom dictionary from model storage.

#### persist

```python
 | persist() -> None
```

Persist the custom dictionaries.

