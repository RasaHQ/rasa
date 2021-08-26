---
sidebar_label: rasa.graph_components.providers.story_graph_provider
title: rasa.graph_components.providers.story_graph_provider
---
## StoryGraphProvider Objects

```python
class StoryGraphProvider(GraphComponent)
```

Provides the training data from stories.

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any]) -> None
```

Creates provider from config.

#### get\_default\_config

```python
 | @staticmethod
 | get_default_config() -> Dict[Text, Any]
```

Returns default configuration (see parent class for full docstring).

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> StoryGraphProvider
```

Creates component (see parent class for full docstring).

#### provide

```python
 | provide(importer: TrainingDataImporter) -> StoryGraph
```

Provides the story graph from the training data.

**Arguments**:

- `importer` - instance of TrainingDataImporter.
  

**Returns**:

  The story graph containing stories and rules used for training.

