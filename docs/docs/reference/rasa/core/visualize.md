---
sidebar_label: rasa.core.visualize
title: rasa.core.visualize
---
#### visualize

```python
def visualize(domain_path: Text, stories_path: Text, nlu_data_path: Text, output_path: Text, max_history: int) -> None
```

Visualizes stories as graph.

**Arguments**:

- `domain_path` - Path to the domain file.
- `stories_path` - Path to the stories files.
- `nlu_data_path` - Path to the NLU training data which can be used to interpolate
  intents with actual examples in the graph.
- `output_path` - Path where the created graph should be persisted.
- `max_history` - Max history to use for the story visualization.

