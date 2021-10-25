---
sidebar_label: rasa.jupyter
title: rasa.jupyter
---
#### pprint

```python
pprint(obj: Any) -> None
```

Prints JSONs with indent.

#### chat

```python
chat(model_path: Optional[Text] = None, endpoints: Optional[Text] = None, agent: Optional["Agent"] = None) -> None
```

Chat to the bot within a Jupyter notebook.

**Arguments**:

- `model_path` - Path to a combined Rasa model.
- `endpoints` - Path to a yaml with the action server is custom actions are defined.
- `agent` - Rasa Core agent (used if no Rasa model given).

