---
sidebar_label: rasa.core.slots
title: rasa.core.slots
---
## Slot Objects

```python
class Slot()
```

#### feature\_dimensionality

```python
 | feature_dimensionality() -> int
```

How many features this single slot creates.

The dimensionality of the array returned by `as_feature` needs
to correspond to this value.

#### add\_default\_value

```python
 | add_default_value() -> None
```

Add a default value to a slots user-defined values

#### has\_features

```python
 | has_features() -> bool
```

Indicate if the slot creates any features.

#### value\_reset\_delay

```python
 | value_reset_delay() -> Optional[int]
```

After how many turns the slot should be reset to the initial_value.

If the delay is set to `None`, the slot will keep its value forever.

#### resolve\_by\_type

```python
 | @staticmethod
 | resolve_by_type(type_name) -> Type["Slot"]
```

Returns a slots class by its type name.

