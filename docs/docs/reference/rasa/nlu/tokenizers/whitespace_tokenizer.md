---
sidebar_label: rasa.nlu.tokenizers.whitespace_tokenizer
title: rasa.nlu.tokenizers.whitespace_tokenizer
---
## WhitespaceTokenizer Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
class WhitespaceTokenizer(Tokenizer)
```

Creates features for entity extraction.

#### not\_supported\_languages

```python
@staticmethod
def not_supported_languages() -> Optional[List[Text]]
```

The languages that are not supported.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

Returns the component&#x27;s default config.

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any]) -> None
```

Initialize the tokenizer.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> WhitespaceTokenizer
```

Creates a new component (see parent class for full docstring).

#### remove\_emoji

```python
def remove_emoji(text: Text) -> Text
```

Remove emoji if the full text, aka token, matches the emoji regex.

