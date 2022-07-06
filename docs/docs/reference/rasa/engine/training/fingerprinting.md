---
sidebar_label: rasa.engine.training.fingerprinting
title: rasa.engine.training.fingerprinting
---
## Fingerprintable Objects

```python
@runtime_checkable
class Fingerprintable(Protocol)
```

Interface that enforces training data can be fingerprinted.

#### fingerprint

```python
def fingerprint() -> Text
```

Returns a unique stable fingerprint of the data.

#### calculate\_fingerprint\_key

```python
def calculate_fingerprint_key(graph_component_class: Type, config: Dict[Text, Any], inputs: Dict[Text, Fingerprintable]) -> Text
```

Calculates a fingerprint key that uniquely represents a single node&#x27;s execution.

**Arguments**:

- `graph_component_class` - The graph component class.
- `config` - The component config.
- `inputs` - The inputs as a mapping of parent node name to input value.
  

**Returns**:

  The fingerprint key.

