---
sidebar_label: rasa.nlu.emulators.wit
title: rasa.nlu.emulators.wit
---
## WitEmulator Objects

```python
class WitEmulator(Emulator)
```

Emulates the response format of this wit.ai endpoint.

More information about the endpoint:
https://wit.ai/docs/http/20200513/#get__message_link

#### normalise\_response\_json

```python
 | normalise_response_json(data: Dict[Text, Any]) -> Dict[Text, Any]
```

Transform response JSON to wit.ai format.

**Arguments**:

- `data` - input JSON data as a dictionary.
  

**Returns**:

  The transformed input data.

