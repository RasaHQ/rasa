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

#### component\_config\_from\_pipeline

```python
component_config_from_pipeline(index: int, pipeline: List[Dict[Text, Any]], defaults: Optional[Dict[Text, Any]] = None) -> Dict[Text, Any]
```

Gets the configuration of the `index`th component.

**Arguments**:

- `index` - Index of the component in the pipeline.
- `pipeline` - Configurations of the components in the pipeline.
- `defaults` - Default configuration.
  

**Returns**:

  The `index`th component configuration, expanded
  by the given defaults.

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

#### override

```python
 | override(config: Optional[Dict[Text, Any]] = None) -> None
```

Overrides default config with given values.

**Arguments**:

- `config` - New values for the configuration.

