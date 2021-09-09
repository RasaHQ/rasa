---
sidebar_label: rasa.nlu.featurizers.dense_featurizer.spacy_featurizer
title: rasa.nlu.featurizers.dense_featurizer.spacy_featurizer
---
## SpacyFeaturizerGraphComponent Objects

```python
class SpacyFeaturizerGraphComponent(DenseFeaturizer2,  GraphComponent)
```

Featurize messages using SpaCy.

#### get\_default\_config

```python
 | @staticmethod
 | get_default_config() -> Dict[Text, Any]
```

The component&#x27;s default config (see parent class for full docstring).

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any], name: Text) -> None
```

Initializes SpacyFeaturizerGraphComponent.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> GraphComponent
```

Creates a new component (see parent class for full docstring).

#### process

```python
 | process(messages: List[Message]) -> List[Message]
```

Processes incoming messages and computes and sets features.

#### validate\_config

```python
 | @classmethod
 | validate_config(cls, config: Dict[Text, Any]) -> None
```

Validates that the component is configured properly.

#### validate\_compatibility\_with\_tokenizer

```python
 | @classmethod
 | validate_compatibility_with_tokenizer(cls, config: Dict[Text, Any], tokenizer_type: Type[Tokenizer]) -> None
```

Validates that the featurizer is compatible with the given tokenizer.

