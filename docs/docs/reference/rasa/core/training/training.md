---
sidebar_label: rasa.core.training.training
title: rasa.core.training.training
---
## ActionFingerprint Objects

```python
@dataclasses.dataclass
class ActionFingerprint()
```

Dataclass to represent an action fingerprint.

#### create\_action\_fingerprints

```python
create_action_fingerprints(trackers: List["DialogueStateTracker"], domain: "Domain") -> Dict[Text, ActionFingerprint]
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

