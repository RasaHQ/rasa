---
sidebar_label: rasa.shared.core.slots
title: rasa.shared.core.slots
---
## InvalidSlotTypeException Objects

```python
class InvalidSlotTypeException(RasaException)
```

Raised if a slot type is invalid.

## InvalidSlotConfigError Objects

```python
class InvalidSlotConfigError(RasaException,  ValueError)
```

Raised if a slot&#x27;s config is invalid.

## Slot Objects

```python
class Slot()
```

Key-value store for storing information during a conversation.

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

#### reset

```python
 | reset() -> None
```

Resets the slot&#x27;s value to the initial value.

#### value

```python
 | @property
 | value() -> Any
```

Gets the slot&#x27;s value.

#### value

```python
 | @value.setter
 | value(value: Any) -> None
```

Sets the slot&#x27;s value.

#### has\_been\_set

```python
 | @property
 | has_been_set() -> bool
```

Indicates if the slot&#x27;s value has been set.

#### resolve\_by\_type

```python
 | @staticmethod
 | resolve_by_type(type_name: Text) -> Type["Slot"]
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

## ListSlot Objects

```python
class ListSlot(Slot)
```

#### value

```python
 | @Slot.value.setter
 | value(value: Any) -> None
```

Sets the slot&#x27;s value.

## UnfeaturizedSlot Objects

```python
class UnfeaturizedSlot(Slot)
```

Deprecated slot type to represent slots which don&#x27;t influence conversations.

#### \_\_init\_\_

```python
 | __init__(name: Text, initial_value: Any = None, value_reset_delay: Optional[int] = None, auto_fill: bool = True, influence_conversation: bool = False) -> None
```

Creates unfeaturized slot.

**Arguments**:

- `name` - The name of the slot.
- `initial_value` - Its initial value.
- `value_reset_delay` - After how many turns the slot should be reset to the
  initial_value. This is behavior is currently not implemented.
- `auto_fill` - `True` if it should be auto-filled by entities with the same
  name.
- `influence_conversation` - `True` if it should be featurized. Only `False`
  is allowed. Any other value will lead to a `InvalidSlotConfigError`.

## CategoricalSlot Objects

```python
class CategoricalSlot(Slot)
```

#### add\_default\_value

```python
 | add_default_value() -> None
```

Adds the special default value to the list of possible values.

#### persistence\_info

```python
 | persistence_info() -> Dict[Text, Any]
```

Returns serialized slot.

## AnySlot Objects

```python
class AnySlot(Slot)
```

Slot which can be used to store any value. Users need to create a subclass of
`Slot` in case the information is supposed to get featurized.

#### \_\_eq\_\_

```python
 | __eq__(other: Any) -> bool
```

Compares object with other object.

