---
sidebar_label: rasa.nlu.tokenizers.mitie_tokenizer
title: rasa.nlu.tokenizers.mitie_tokenizer
---
## MitieTokenizerGraphComponent Objects

```python
class MitieTokenizerGraphComponent(TokenizerGraphComponent)
```

Tokenizes messages using the `mitie` library..

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

Returns default config (see parent class for full docstring).

#### required\_packages

```python
@staticmethod
def required_packages() -> List[Text]
```

Any extra python dependencies required for this component to run.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> MitieTokenizerGraphComponent
```

Creates a new component (see parent class for full docstring).

#### tokenize

```python
def tokenize(message: Message, attribute: Text) -> List[Token]
```

Tokenizes the text of the provided attribute of the incoming message.

