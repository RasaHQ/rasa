---
sidebar_label: rasa.nlu.emulators.emulator
title: rasa.nlu.emulators.emulator
---
## Emulator Objects

```python
class Emulator()
```

Emulator specifies how requests and responses are getting transformed.

#### name

```python
 | @classmethod
 | name(cls) -> Text
```

Name that identifies the emulator.

#### normalise\_request\_json

```python
 | normalise_request_json(data: Dict[Text, Any]) -> Dict[Text, Any]
```

Transform request JSON to target format.

**Arguments**:

- `data` - input JSON data as a dictionary.
  

**Returns**:

  The transformed input data.

#### normalise\_response\_json

```python
 | normalise_response_json(data: Dict[Text, Any]) -> Dict[Text, Any]
```

Transform response JSON to target format.

**Arguments**:

- `data` - input JSON data as a dictionary.
  

**Returns**:

  The transformed input data.

#### \_\_str\_\_

```python
 | __str__() -> Text
```

Return the string representation of the emulator.

