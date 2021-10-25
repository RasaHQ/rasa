---
sidebar_label: rasa.engine.validation
title: rasa.engine.validation
---
## ParameterInfo Objects

```python
@dataclasses.dataclass
class ParameterInfo()
```

Stores metadata about a function parameter.

#### validate

```python
validate(model_configuration: GraphModelConfiguration) -> None
```

Validates a graph schema.

This tries to validate that the graph structure is correct (e.g. all nodes pass the
correct things into each other) as well as validates the individual graph
components.

**Arguments**:

- `model_configuration` - The model configuration (schemas, language, etc.)
  

**Raises**:

- `GraphSchemaValidationException` - If the validation failed.

