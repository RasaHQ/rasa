---
sidebar_label: rasa.shared.core.slot_mappings
title: rasa.shared.core.slot_mappings
---
## SlotMapping Objects

```python
class SlotMapping(Enum)
```

Defines the available slot mappings.

#### \_\_str\_\_

```python
 | __str__() -> Text
```

Returns a string representation of the object.

#### validate

```python
 | @staticmethod
 | validate(mapping: Dict[Text, Any], slot_name: Text) -> None
```

Validates a slot mapping.

**Arguments**:

- `mapping` - The mapping which is validated.
- `slot_name` - The name of the slot which is mapped by this mapping.
  

**Raises**:

- `InvalidDomain` - In case the slot mapping is not valid.

#### intent\_is\_desired

```python
 | @staticmethod
 | intent_is_desired(mapping: Dict[Text, Any], tracker: "DialogueStateTracker", domain: "Domain") -> bool
```

Checks whether user intent matches slot mapping intent specifications.

#### to\_list

```python
 | @staticmethod
 | to_list(x: Optional[Any]) -> List[Any]
```

Convert object to a list if it isn&#x27;t.

#### entity\_is\_desired

```python
 | @staticmethod
 | entity_is_desired(mapping: Dict[Text, Any], tracker: "DialogueStateTracker") -> bool
```

Checks whether slot should be filled by an entity in the input or not.

**Arguments**:

- `mapping` - Slot mapping.
- `tracker` - The tracker.
  

**Returns**:

  True, if slot should be filled, false otherwise.

#### check\_mapping\_validity

```python
 | @staticmethod
 | check_mapping_validity(slot_name: Text, mapping: Dict[Text, Any], domain: "Domain") -> bool
```

Checks the mapping for validity.

**Arguments**:

- `mapping` - Slot mapping.
- `domain` - The domain to check against.
  

**Returns**:

  True, if intent and entity specified in a mapping exist in domain.

#### validate\_slot\_mappings

```python
validate_slot_mappings(domain_slots: Dict[Text, Any]) -> None
```

Raises InvalidDomain exception if slot mappings are invalid.

