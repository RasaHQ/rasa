---
sidebar_label: rasa.nlu.tokenizers._whitespace_tokenizer
title: rasa.nlu.tokenizers._whitespace_tokenizer
---
## WhitespaceTokenizer Objects

```python
class WhitespaceTokenizer(Tokenizer)
```

#### \_\_init\_\_

```python
def __init__(component_config: Dict[Text, Any] = None) -> None
```

Construct a new tokenizer using the WhitespaceTokenizer framework.

#### remove\_emoji

```python
def remove_emoji(text: Text) -> Text
```

Remove emoji if the full text, aka token, matches the emoji regex.

