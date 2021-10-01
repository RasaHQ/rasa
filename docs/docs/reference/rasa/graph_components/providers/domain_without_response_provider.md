---
sidebar_label: rasa.graph_components.providers.domain_without_response_provider
title: rasa.graph_components.providers.domain_without_response_provider
---
## DomainWithoutResponsesProvider Objects

```python
class DomainWithoutResponsesProvider(GraphComponent)
```

Provides domain without information about responses.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> DomainWithoutResponsesProvider
```

Creates component (see parent class for full docstring).

#### provide

```python
def provide(domain: Domain) -> Domain
```

Recreates the given domain but acts as if responses have not been specified.

**Arguments**:

- `domain` - A domain.
  

**Returns**:

  Domain that has been created from the same parameters as the given domain
  but with an empty set of responses.

