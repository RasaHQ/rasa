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
def __init__(cache: TrainingCache, model_storage: ModelStorage)
```

Initializes a `TrainingHook`.

**Arguments**:

- `cache` - Cache used to store fingerprints and outputs.
- `model_storage` - Used to cache `Resource`s.

#### on\_before\_node

```python
def on_before_node(node_name: Text, execution_context: ExecutionContext, config: Dict[Text, Any], received_inputs: Dict[Text, Any]) -> Dict
```

Calculates the run fingerprint for use in `on_after_node`.

#### on\_after\_node

```python
def on_after_node(node_name: Text, execution_context: ExecutionContext, config: Dict[Text, Any], output: Any, input_hook_data: Dict) -> None
```

Stores the fingerprints and caches the output of the node.

