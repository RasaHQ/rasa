---
sidebar_label: rasa.engine.runner.dask
title: rasa.engine.runner.dask
---
## DaskGraphRunner Objects

```python
class DaskGraphRunner(GraphRunner)
```

Dask implementation of a `GraphRunner`.

#### \_\_init\_\_

```python
 | __init__(graph_schema: GraphSchema, model_storage: ModelStorage, execution_context: ExecutionContext, hooks: Optional[List[GraphNodeHook]] = None) -> None
```

Initializes a `DaskGraphRunner`.

**Arguments**:

- `graph_schema` - The graph schema that will be run.
- `model_storage` - Storage which graph components can use to persist and load
  themselves.
- `execution_context` - Information about the current graph run to be passed to
  each node.
- `hooks` - These are called before and after the execution of each node.

#### create

```python
 | @classmethod
 | create(cls, graph_schema: GraphSchema, model_storage: ModelStorage, execution_context: ExecutionContext, hooks: Optional[List[GraphNodeHook]] = None) -> DaskGraphRunner
```

Creates the runner (see parent class for full docstring).

#### run

```python
 | run(inputs: Optional[Dict[Text, Any]] = None, targets: Optional[List[Text]] = None) -> Dict[Text, Any]
```

Runs the graph (see parent class for full docstring).

