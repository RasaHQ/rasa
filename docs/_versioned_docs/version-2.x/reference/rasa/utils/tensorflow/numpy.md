---
sidebar_label: rasa.utils.tensorflow.numpy
title: rasa.utils.tensorflow.numpy
---
#### values\_to\_numpy

```python
values_to_numpy(data: Optional[Dict[Any, Any]]) -> Optional[Dict[Any, Any]]
```

Replaces all tensorflow-tensor values with their numpy versions.

**Arguments**:

- `data` - Any dictionary for which values should be converted.
  

**Returns**:

  A dictionary identical to `data` except that tensor values are
  replaced by their corresponding numpy arrays.

