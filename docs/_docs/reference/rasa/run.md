---
sidebar_label: rasa.run
title: rasa.run
---
#### run

```python
run(model: Text, endpoints: Text, connector: Text = None, credentials: Text = None, **kwargs: Dict, ,)
```

Runs a Rasa model.

**Arguments**:

- `model` - Path to model archive.
- `endpoints` - Path to endpoints file.
- `connector` - Connector which should be use (overwrites `credentials`
  field).
- `credentials` - Path to channel credentials file.
- `**kwargs` - Additional arguments which are passed to
  `rasa.core.run.serve_application`.

