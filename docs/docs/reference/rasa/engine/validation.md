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
def validate(schema: GraphSchema, language: Optional[Text], is_train_graph: bool) -> None
```

Validates a graph schema.

This tries to validate that the graph structure is correct (e.g. all nodes pass the
correct things into each other) as well as validates the individual graph
components.

**Arguments**:

- `schema` - The schema which needs validating.
- `language` - Used to validate if all components support the language the assistant
  is used in. If the language is `None`, all components are assumed to be
  compatible.
- `is_train_graph` - Whether the graph is used for training.
  

**Raises**:

- `GraphSchemaValidationException` - If the validation failed.

