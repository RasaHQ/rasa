---
sidebar_label: rasa.engine.training.hooks
title: rasa.engine.training.hooks
---
## TrainingHook Objects

```python
class TrainingHook(GraphNodeHook)
```

Caches fingerprints and outputs of nodes during model training.

#### \_\_init\_\_

```python
 | __init__(cache: TrainingCache, model_storage: ModelStorage, pruned_schema: GraphSchema) -> None
```

Initializes a `TrainingHook`.

**Arguments**:

- `cache` - Cache used to store fingerprints and outputs.
- `model_storage` - Used to cache `Resource`s.
- `pruned_schema` - The pruned training schema.

#### on\_before\_node

```python
 | on_before_node(node_name: Text, execution_context: ExecutionContext, config: Dict[Text, Any], received_inputs: Dict[Text, Any]) -> Dict
```

Calculates the run fingerprint for use in `on_after_node`.

#### on\_after\_node

```python
 | on_after_node(node_name: Text, execution_context: ExecutionContext, config: Dict[Text, Any], output: Any, input_hook_data: Dict) -> None
```

Stores the fingerprints and caches the output of the node.

## LoggingHook Objects

```python
class LoggingHook(GraphNodeHook)
```

Logs the training of components.

#### on\_before\_node

```python
 | on_before_node(node_name: Text, execution_context: ExecutionContext, config: Dict[Text, Any], received_inputs: Dict[Text, Any]) -> Dict
```

Logs the training start of a graph node.

#### on\_after\_node

```python
 | on_after_node(node_name: Text, execution_context: ExecutionContext, config: Dict[Text, Any], output: Any, input_hook_data: Dict) -> None
```

Logs when a component finished its training.

