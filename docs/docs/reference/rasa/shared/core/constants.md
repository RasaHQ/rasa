---
sidebar_label: rasa.shared.core.constants
title: rasa.shared.core.constants
---
## SlotMappingType Objects

```python
class SlotMappingType(Enum)
```

Slot mapping types.

#### \_\_str\_\_

```python
 | __str__() -> str
```

Returns the string representation that should be used in config files.

#### is\_predefined\_type

```python
 | is_predefined_type() -> bool
```

Returns True iff the mapping type is predefined.

That is, to evaluate the mapping no custom action execution is needed.

