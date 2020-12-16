---
sidebar_label: config
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

#### component\_config\_from\_pipeline

```python
component_config_from_pipeline(index: int, pipeline: List[Dict[Text, Any]], defaults: Optional[Dict[Text, Any]] = None) -> Dict[Text, Any]
```

Get config of the component with the given index in the pipeline.

**Arguments**:

- `index` - index the component in the pipeline
- `pipeline` - a list of component configs in the NLU pipeline
- `defaults` - default config of the component
  

**Returns**:

  config of the component

## RasaNLUModelConfig Objects

```python
class RasaNLUModelConfig()
```

A class that stores NLU model configuration parameters.

#### \_\_init\_\_

```python
 | __init__(configuration_values: Optional[Dict[Text, Any]] = None) -> None
```

Create a model configuration.

**Arguments**:

- `configuration_values` - optional dictionary to override defaults.

