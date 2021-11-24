---
sidebar_label: rasa.graph_components.validators.default_recipe_validator
title: rasa.graph_components.validators.default_recipe_validator
---
## DefaultV1RecipeValidator Objects

```python
class DefaultV1RecipeValidator(GraphComponent)
```

Validates a &quot;DefaultV1&quot; configuration against the training data and domain.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> DefaultV1RecipeValidator
```

Creates a new `ConfigValidator` (see parent class for full docstring).

#### \_\_init\_\_

```python
 | __init__(graph_schema: GraphSchema) -> None
```

Instantiates a new `ConfigValidator`.

**Arguments**:

- `graph_schema` - a graph schema

#### validate

```python
 | validate(importer: TrainingDataImporter) -> TrainingDataImporter
```

Validates the current graph schema against the training data and domain.

**Arguments**:

- `importer` - the training data importer which can also load the domain

**Raises**:

  `InvalidConfigException` or `InvalidDomain` in case there is some mismatch

