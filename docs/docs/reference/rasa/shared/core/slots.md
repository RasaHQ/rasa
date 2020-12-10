---
sidebar_label: slots
title: rasa.shared.core.slots
---

## InvalidSlotTypeException Objects

```python
class InvalidSlotTypeException(RasaException)
```

Raised if a slot type is invalid.

## Slot Objects

```python
class Slot()
```

#### \_\_init\_\_

```python
 | __init__(name: Text, initial_value: Any = None, value_reset_delay: Optional[int] = None, auto_fill: bool = True, influence_conversation: bool = True) -> None
```

Create a Slot.

**Arguments**:

- `name` - The name of the slot.
- `initial_value` - The initial value of the slot.
- `value_reset_delay` - After how many turns the slot should be reset to the
  initial_value. This is behavior is currently not implemented.
- `auto_fill` - `True` if the slot should be filled automatically by entities
  with the same name.
- `influence_conversation` - If `True` the slot will be featurized and hence
  influence the predictions of the dialogue polices.

#### feature\_dimensionality

```python
 | feature_dimensionality() -> int
```

How many features this single slot creates.

**Returns**:

  The number of features. `0` if the slot is unfeaturized. The dimensionality
  of the array returned by `as_feature` needs to correspond to this value.

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

## FloatSlot Objects

```python
class FloatSlot(Slot)
```

#### persistence\_info

```python
 | persistence_info() -> Dict[Text, Any]
```

Returns relevant information to persist this slot.

## BooleanSlot Objects

```python
class BooleanSlot(Slot)
```

A slot storing a truth value.

#### bool\_from\_any

```python
bool_from_any(x: Any) -> bool
```

Converts bool/float/int/str to bool or raises error

## AnySlot Objects

```python
class AnySlot(Slot)
```

Slot which can be used to store any value. Users need to create a subclass of
`Slot` in case the information is supposed to get featurized.

