---
sidebar_label: rasa.core.conversation
title: rasa.core.conversation
---
## Dialogue Objects

```python
class Dialogue()
```

A dialogue comprises a list of Turn objects

#### \_\_init\_\_

```python
 | __init__(name: Text, events: List["Event"]) -> None
```

This function initialises the dialogue with the dialogue name and the event
list.

#### \_\_str\_\_

```python
 | __str__() -> Text
```

This function returns the dialogue and turns.

#### as\_dict

```python
 | as_dict() -> Dict
```

This function returns the dialogue as a dictionary to assist in
serialization.

#### from\_parameters

```python
 | @classmethod
 | from_parameters(cls, parameters: Dict[Text, Any]) -> "Dialogue"
```

Create `Dialogue` from parameters.

**Arguments**:

- `parameters` - Serialised dialogue, should contain keys &#x27;name&#x27; and &#x27;events&#x27;.
  

**Returns**:

  Deserialised `Dialogue`.

