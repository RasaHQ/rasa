---
sidebar_label: rasa.nlu.emulators.luis
title: rasa.nlu.emulators.luis
---
## LUISEmulator Objects

```python
class LUISEmulator(Emulator)
```

Emulates the response format of the LUIS Endpoint API v3.0 /predict endpoint.

https://westcentralus.dev.cognitive.microsoft.com/docs/services/luis-endpoint-api-v3-0/
https://docs.microsoft.com/en-us/azure/cognitive-services/LUIS/luis-concept-data-extraction?tabs=V3

#### normalise\_response\_json

```python
 | normalise_response_json(data: Dict[Text, Any]) -> Dict[Text, Any]
```

Transform response JSON to LUIS format.

**Arguments**:

- `data` - input JSON data as a dictionary.
  

**Returns**:

  The transformed input data.

