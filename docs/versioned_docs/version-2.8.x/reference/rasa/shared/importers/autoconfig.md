---
sidebar_label: rasa.shared.importers.autoconfig
title: rasa.shared.importers.autoconfig
---
#### get\_configuration

```python
get_configuration(config_file_path: Optional[Text], training_type: Optional[TrainingType] = TrainingType.BOTH) -> Dict[Text, Any]
```

Determine configuration from a configuration file.

Keys that are provided and have a value in the file are kept. Keys that are not
provided are configured automatically.

**Arguments**:

- `config_file_path` - The path to the configuration file.
- `training_type` - NLU, CORE or BOTH depending on what is trained.

