---
sidebar_label: rasa.engine.training.components
title: rasa.engine.training.components
---
## PrecomputedValueProvider Objects

```python
class PrecomputedValueProvider(GraphComponent)
```

Holds the precomputed values of a `GraphNode` from a previous training.

Pre-computed values can either be
- values loaded from cache
- values which were provided during the fingerprint run by input nodes

#### \_\_init\_\_

```python
def __init__(output: Cacheable)
```

Initializes a `PrecomputedValueProvider`.

**Arguments**:

- `output` - The precomputed output to return.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> PrecomputedValueProvider
```

Creates instance (see parent class for full docstring).

#### get\_value

```python
def get_value() -> Cacheable
```

Returns the precomputed output.

#### replace\_schema\_node

```python
@classmethod
def replace_schema_node(cls, node: SchemaNode, output: Any) -> None
```

Updates a `SchemaNode` to use a `PrecomputedValueProvider`.

This is for when we want to use the precomputed output value of a node from a
previous training in a subsequent training. We replace the class in the `uses`
of the node to a be a `PrecomputedValueProvider` configured to return the
precomputed value.

**Arguments**:

- `node` - The node to update.
- `output` - precomputed cached output that the `PrecomputedValueProvider` will
  return.

## FingerprintStatus Objects

```python
@dataclasses.dataclass
class FingerprintStatus()
```

Holds the output of a `FingerprintComponent` and is used to prune the graph.

**Attributes**:

- `output_fingerprint` - A fingerprint of the node&#x27;s output value.
- `is_hit` - `True` if node&#x27;s fingerprint key exists in the cache, `False` otherwise.

#### fingerprint

```python
def fingerprint() -> Text
```

Returns the internal fingerprint.

If there is no fingerprint returns a random string that will never match.

## FingerprintComponent Objects

```python
class FingerprintComponent(GraphComponent)
```

Replaces non-input nodes during a fingerprint run.

#### \_\_init\_\_

```python
def __init__(cache: TrainingCache, config_of_replaced_component: Dict[Text, Any], class_of_replaced_component: Type) -> None
```

Initializes a `FingerprintComponent`.

**Arguments**:

- `cache` - Training cache used to determine if the run is a hit or not.
- `config_of_replaced_component` - Needed to generate the fingerprint key.
- `class_of_replaced_component` - Needed to generate the fingerprint key.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> FingerprintComponent
```

Creates a `FingerprintComponent` (see parent class for full docstring).

#### run

```python
def run(**kwargs: Any) -> FingerprintStatus
```

Calculates the fingerprint key to determine if cached output can be used.

If the fingerprint key matches an entry in the cache it means that there has
been a previous node execution which matches the same component class, component
config and input values. This means that we can potentially prune this node
from the schema, or replace it with a cached value before the next graph run.

**Arguments**:

- `**kwargs` - Inputs from all parent nodes.
  

**Returns**:

  A `FingerprintStatus` determining if the run was a hit, and if it was a hit
  also the output fingerprint from the cache.

#### replace\_schema\_node

```python
@classmethod
def replace_schema_node(cls, node: SchemaNode, cache: TrainingCache) -> None
```

Updates a `SchemaNode` to use a `FingerprintComponent`.

This is for when we want to do a fingerprint run. During the fingerprint run we
replace all non-input nodes with `FingerprintComponent`s so we can determine
whether they are able to be pruned or cached before the next graph run without
running the actual components.


**Arguments**:

- `node` - The node to update.
- `cache` - The cache is needed to determine of there is cache hit for the
  fingerprint key.

