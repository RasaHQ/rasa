---
sidebar_label: rasa.engine.training.graph_trainer
title: rasa.engine.training.graph_trainer
---
## GraphTrainer Objects

```python
class GraphTrainer()
```

Trains a model using a graph schema.

#### \_\_init\_\_

```python
def __init__(model_storage: ModelStorage, cache: TrainingCache, graph_runner_class: Type[GraphRunner]) -> None
```

Initializes a `GraphTrainer`.

**Arguments**:

- `model_storage` - Storage which graph components can use to persist and load.
  Also used for packaging the trained model.
- `cache` - Cache used to store fingerprints and outputs.
- `graph_runner_class` - The class to instantiate the runner from.

#### train

```python
def train(train_schema: GraphSchema, predict_schema: GraphSchema, domain_path: Path, output_filename: Path) -> GraphRunner
```

Trains and packages a model and returns the prediction graph runner.

**Arguments**:

- `train_schema` - The train graph schema.
- `predict_schema` - The predict graph schema.
- `domain_path` - The path to the domain file.
- `output_filename` - The location to save the packaged model.
  

**Returns**:

  A graph runner loaded with the predict schema.

