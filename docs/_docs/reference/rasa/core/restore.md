---
sidebar_label: rasa.core.restore
title: rasa.core.restore
---
#### align\_lists

```python
align_lists(predictions: List[Text], golds: List[Text]) -> Tuple[List[Text], List[Text]]
```

Align two lists trying to keep same elements at the same index.

If lists contain different items at some indices, the algorithm will
try to find the best alignment and pad with `None`
values where necessary.

#### actions\_since\_last\_utterance

```python
actions_since_last_utterance(tracker: DialogueStateTracker) -> List[Text]
```

Extract all events after the most recent utterance from the user.

#### replay\_events

```python
async replay_events(tracker: DialogueStateTracker, agent: "Agent") -> None
```

Take a tracker and replay the logged user utterances against an agent.

During replaying of the user utterances, the executed actions and events
created by the agent are compared to the logged ones of the tracker that
is getting replayed. If they differ, a warning is logged.

At the end, the tracker stored in the agent&#x27;s tracker store for the
same sender id will have quite the same state as the one
that got replayed.

#### load\_tracker\_from\_json

```python
load_tracker_from_json(tracker_dump: Text, domain: Domain) -> DialogueStateTracker
```

Read the json dump from the file and instantiate a tracker it.

