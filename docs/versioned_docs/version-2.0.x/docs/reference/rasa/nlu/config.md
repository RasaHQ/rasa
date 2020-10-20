---
sidebar_label: rasa.nlu.config
title: rasa.nlu.config
---

## InvalidConfigError Objects

```python
class InvalidConfigError(ValueError,  RasaException)
```

Raised if an invalid configuration is encountered.

## RasaNLUModelConfig Objects

```python
class RasaNLUModelConfig()
```

#### \_\_init\_\_

```python
 | __init__(configuration_values: Optional[Dict[Text, Any]] = None) -> None
```

Create a model configuration, optionally overriding
defaults with a dictionary ``configuration_values``.

