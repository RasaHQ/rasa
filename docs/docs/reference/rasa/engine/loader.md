---
sidebar_label: rasa.engine.loader
title: rasa.engine.loader
---
#### load\_predict\_graph\_runner

```python
def load_predict_graph_runner(storage_path: Path, model_archive_path: Path, model_storage_class: Type[ModelStorage], graph_runner_class: Type[GraphRunner]) -> Tuple[ModelMetadata, GraphRunner]
```

Loads a model from an archive and creates the prediction graph runner.

**Arguments**:

- `storage_path` - Directory which contains the persisted graph components.
- `model_archive_path` - The path to the model archive.
- `model_storage_class` - The class to instantiate the model storage from.
- `graph_runner_class` - The class to instantiate the runner from.
  

**Returns**:

  A tuple containing the model metadata and the prediction graph runner.

