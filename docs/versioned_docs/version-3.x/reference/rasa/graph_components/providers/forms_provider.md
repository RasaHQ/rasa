---
sidebar_label: rasa.graph_components.providers.forms_provider
title: rasa.graph_components.providers.forms_provider
---
## Forms Objects

```python
@dataclasses.dataclass
class Forms()
```

Holds the forms of the domain.

#### fingerprint

```python
def fingerprint() -> Text
```

Returns a fingerprint of the responses.

#### get

```python
def get(key: Text, default: Any) -> Any
```

Returns the value for the given key.

## FormsProvider Objects

```python
class FormsProvider(GraphComponent)
```

Provides forms during training and inference time.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage,
           resource: Resource,
           execution_context: ExecutionContext) -> FormsProvider
```

Creates component (see parent class for full docstring).

#### provide

```python
def provide(domain: Domain) -> Forms
```

Returns the forms from the given domain.

