---
sidebar_label: rasa.graph_components.providers.rule_only_provider
title: rasa.graph_components.providers.rule_only_provider
---
## RuleOnlyDataProvider Objects

```python
@dataclasses.dataclass
class RuleOnlyDataProvider(GraphComponent)
```

Provides slots and loops that are only used in rules to other policies.

Policies can use this to exclude features which are only used by rules from the
featurization.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> RuleOnlyDataProvider
```

Creates component (see parent class for docstring).

#### provide

```python
def provide() -> Dict[Text, Any]
```

Provides data to other graph component.

