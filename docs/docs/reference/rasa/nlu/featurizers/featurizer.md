---
sidebar_label: rasa.nlu.featurizers.featurizer
title: rasa.nlu.featurizers.featurizer
---
## Featurizer2 Objects

```python
class Featurizer2(Generic[FeatureType],  ABC)
```

Base class for all featurizers.

#### get\_default\_config

```python
 | @staticmethod
 | get_default_config() -> Dict[Text, Any]
```

Returns the component&#x27;s default config.

#### \_\_init\_\_

```python
 | __init__(name: Text, config: Dict[Text, Any]) -> None
```

Instantiates a new featurizer.

**Arguments**:

- `config` - configuration
- `name` - a name that can be used as identifier, in case the configuration does
  not specify an `alias` (or this `alias` is None)

#### validate\_config

```python
 | @classmethod
 | @abstractmethod
 | validate_config(cls, config: Dict[Text, Any]) -> None
```

Validates that the component is configured properly.

#### validate\_compatibility\_with\_tokenizer

```python
 | @classmethod
 | @abstractmethod
 | validate_compatibility_with_tokenizer(cls, config: Dict[Text, Any], tokenizer_type: Type[Tokenizer]) -> None
```

Validates that the featurizer is compatible with the given tokenizer.

#### add\_features\_to\_message

```python
 | add_features_to_message(sequence: FeatureType, sentence: Optional[FeatureType], attribute: Text, message: Message) -> None
```

Adds sequence and sentence features for the attribute to the given message.

**Arguments**:

- `sequence` - sequence feature matrix
- `sentence` - sentence feature matrix
- `attribute` - the attribute which both features describe
- `message` - the message to which we want to add those features

#### validate\_configs\_compatible

```python
 | @staticmethod
 | validate_configs_compatible(featurizer_configs: Iterable[Dict[Text, Any]]) -> None
```

Validates that the given configurations of featurizers can be used together.

**Raises**:

  `InvalidConfigException` if the given featurizers should not be used in
  the same graph.

