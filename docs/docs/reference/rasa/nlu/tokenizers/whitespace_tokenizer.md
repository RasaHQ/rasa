---
sidebar_label: rasa.nlu.tokenizers.whitespace_tokenizer
title: rasa.nlu.tokenizers.whitespace_tokenizer
---
## WhitespaceTokenizerGraphComponent Objects

```python
class WhitespaceTokenizerGraphComponent(TokenizerGraphComponent)
```

Creates features for entity extraction.

#### not\_supported\_languages

```python
 | @staticmethod
 | not_supported_languages() -> Optional[List[Text]]
```

The languages that are not supported.

#### get\_default\_config

```python
 | @staticmethod
 | get_default_config() -> Dict[Text, Any]
```

Returns the component&#x27;s default config.

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any]) -> None
```

Initialize the tokenizer.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> "WhitespaceTokenizerGraphComponent"
```

Creates a new component (see parent class for full docstring).

#### remove\_emoji

```python
 | remove_emoji(text: Text) -> Text
```

Remove emoji if the full text, aka token, matches the emoji regex.

