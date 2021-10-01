---
sidebar_label: rasa.core.training.training
title: rasa.core.training.training
---
#### create\_action\_fingerprints

```python
def create_action_fingerprints(trackers: List["DialogueStateTracker"], domain: "Domain") -> Dict[Text, Dict[Text, List[Text]]]
```

Fingerprint each action using the events it created during train.

This allows us to emit warnings when the model is used
if an action does things it hasn&#x27;t done during training,
or if rules are incomplete.

**Arguments**:

- `trackers` - the list of trackers
- `domain` - the domain
  

**Returns**:

  a nested dictionary of action names and slots and active loops
  that this action sets

