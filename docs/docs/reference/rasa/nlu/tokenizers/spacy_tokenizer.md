---
sidebar_label: rasa.nlu.tokenizers.spacy_tokenizer
title: rasa.nlu.tokenizers.spacy_tokenizer
---
## SpacyTokenizer Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
class SpacyTokenizer(Tokenizer)
```

Tokenizer that uses SpaCy.

#### required\_components

```python
@classmethod
def required_components(cls) -> List[Type]
```

Components that should be included in the pipeline before this component.

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

