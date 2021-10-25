---
sidebar_label: rasa.graph_components.providers.training_tracker_provider
title: rasa.graph_components.providers.training_tracker_provider
---
## TrainingTrackerProvider Objects

```python
class TrainingTrackerProvider(GraphComponent)
```

Provides training trackers to policies based on training stories.

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any]) -> None
```

Creates provider from config.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

Returns default configuration (see parent class for full docstring).

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> TrainingTrackerProvider
```

Creates component (see parent class for full docstring).

#### provide

```python
def provide(story_graph: StoryGraph, domain: Domain) -> List[TrackerWithCachedStates]
```

Generates the training trackers from the training data.

**Arguments**:

- `story_graph` - The story graph containing the test stories and rules.
- `domain` - The domain of the model.
  

**Returns**:

  The trackers which can be used to train dialogue policies.

