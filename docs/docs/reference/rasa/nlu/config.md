---
sidebar_label: rasa.nlu.config
title: rasa.nlu.config
---

#### load

```python
load(config: Optional[Union[Text, Dict]] = None, **kwargs: Any) -> "RasaNLUModelConfig"
```

Create configuration from file or dict.

**Arguments**:

- `config` - a file path, a dictionary with configuration keys. If set to
  `None` the configuration will be loaded from the default file
  path.
  

**Returns**:

  Configuration object.

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

