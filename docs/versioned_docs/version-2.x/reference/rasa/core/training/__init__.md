---
sidebar_label: training
title: rasa.core.training
---

#### load\_data

```python
async load_data(resource_name: Union[Text, "TrainingDataImporter"], domain: "Domain", remove_duplicates: bool = True, unique_last_num_states: Optional[int] = None, augmentation_factor: int = 50, tracker_limit: Optional[int] = None, use_story_concatenation: bool = True, debug_plots: bool = False, exclusion_percentage: Optional[int] = None) -> List["TrackerWithCachedStates"]
```

Load training data from a resource.

**Arguments**:

- `resource_name` - resource to load the data from. either a path or an importer
- `domain` - domain used for loading
- `remove_duplicates` - should duplicated training examples be removed?
- `unique_last_num_states` - number of states in a conversation that make the
  a tracker unique (this is used to identify duplicates)
  augmentation_factor:
  by how much should the story training data be augmented
  tracker_limit:
  maximum number of trackers to generate during augmentation
  use_story_concatenation:
  should stories be concatenated when doing data augmentation
  debug_plots:
  generate debug plots during loading
  exclusion_percentage:
  how much data to exclude
  

**Returns**:

  list of loaded trackers

#### persist\_data

```python
persist_data(trackers: List["DialogueStateTracker"], path: Text) -> None
```

Dump a list of dialogue trackers in the story format to disk.

