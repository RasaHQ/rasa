---
sidebar_label: rasa.nlu.emulators.dialogflow
title: rasa.nlu.emulators.dialogflow
---
## DialogflowEmulator Objects

```python
class DialogflowEmulator(Emulator)
```

Emulates the response format of the DialogFlow.

__noqa: W505__

https://cloud.google.com/dialogflow/es/docs/reference/rest/v2/projects.agent.environments.users.sessions/detectIntent
https://cloud.google.com/dialogflow/es/docs/reference/rest/v2/DetectIntentResponse

#### normalise\_response\_json

```python
def normalise_response_json(data: Dict[Text, Any]) -> Dict[Text, Any]
```

&quot;Transform response JSON to DialogFlow format.

**Arguments**:

- `data` - input JSON data as a dictionary.
  

**Returns**:

  The transformed input data.

