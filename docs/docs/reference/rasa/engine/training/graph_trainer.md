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
def train(model_configuration: GraphModelConfiguration, importer: TrainingDataImporter, output_filename: Path, force_retraining: bool = False, is_finetuning: bool = False) -> ModelMetadata
```

Trains and packages a model and returns the prediction graph runner.

**Arguments**:

- `model_configuration` - The model configuration (schemas, language, etc.)
- `importer` - The importer which provides the training data for the training.
- `output_filename` - The location to save the packaged model.
- `force_retraining` - If `True` then the cache is skipped and all components
  are retrained.
  

**Returns**:

  The metadata describing the trained model.

#### fingerprint

```python
def fingerprint(train_schema: GraphSchema, importer: TrainingDataImporter, is_finetuning: bool = False) -> Dict[Text, Union[FingerprintStatus, Any]]
```

Runs the graph using fingerprints to determine which nodes need to re-run.

Nodes which have a matching fingerprint key in the cache can either be removed
entirely from the graph, or replaced with a cached value if their output is
needed by descendent nodes.

**Arguments**:

- `train_schema` - The train graph schema that will be run in fingerprint mode.
- `importer` - The importer which provides the training data for the training.
- `is_finetuning` - `True` if we want to finetune the model.
  

**Returns**:

  Mapping of node names to fingerprint results.

