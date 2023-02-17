---
sidebar_label: rasa.graph_components.providers.responses_provider
title: rasa.graph_components.providers.responses_provider
---
## Responses Objects

```python
@dataclasses.dataclass
class Responses()
```

Holds the responses of the domain.

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

## ResponsesProvider Objects

```python
class ResponsesProvider(GraphComponent)
```

Provides responses during training and inference time.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage,
           resource: Resource,
           execution_context: ExecutionContext) -> ResponsesProvider
```

Creates component (see parent class for full docstring).

#### provide

```python
def provide(domain: Domain) -> Responses
```

Returns the responses from the given domain.

