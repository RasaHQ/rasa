---
sidebar_label: rasa.jupyter
title: rasa.jupyter
---
#### pprint

```python
def pprint(obj: Any) -> None
```

Prints JSONs with indent.

#### chat

```python
def chat(model_path: Optional[Text] = None, endpoints: Optional[Text] = None, agent: Optional["Agent"] = None, interpreter: Optional[NaturalLanguageInterpreter] = None) -> None
```

Chat to the bot within a Jupyter notebook.

**Arguments**:

- `model_path` - Path to a combined Rasa model.
- `endpoints` - Path to a yaml with the action server is custom actions are defined.
- `agent` - Rasa Core agent (used if no Rasa model given).
- `interpreter` - Rasa NLU interpreter (used with Rasa Core agent if no
  Rasa model is given).

