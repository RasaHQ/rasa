---
sidebar_label: rasa.nlu.tokenizers.spacy_tokenizer
title: rasa.nlu.tokenizers.spacy_tokenizer
---
## SpacyTokenizerGraphComponent Objects

```python
class SpacyTokenizerGraphComponent(TokenizerGraphComponent)
```

Tokenizer that uses SpaCy.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

The component&#x27;s default config (see parent class for full docstring).

#### required\_packages

```python
@staticmethod
def required_packages() -> List[Text]
```

Any extra python dependencies required for this component to run.

#### tokenize

```python
def tokenize(message: Message, attribute: Text) -> List[Token]
```

Tokenizes the text of the provided attribute of the incoming message.

