---
sidebar_label: rasa.graph_components.providers.nlu_training_data_provider
title: rasa.graph_components.providers.nlu_training_data_provider
---
## NLUTrainingDataProvider Objects

```python
class NLUTrainingDataProvider(GraphComponent)
```

Provides NLU training data during training.

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource) -> None
```

Creates a new NLU training data provider.

#### get\_default\_config

```python
 | @classmethod
 | get_default_config(cls) -> Dict[Text, Any]
```

Returns the default config for NLU training data provider.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> NLUTrainingDataProvider
```

Creates a new NLU training data provider.

#### provide

```python
 | provide(importer: TrainingDataImporter) -> TrainingData
```

Provides nlu training data during training.

