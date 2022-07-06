---
sidebar_label: rasa.shared.core.training_data.loading
title: rasa.shared.core.training_data.loading
---
#### load\_data\_from\_resource

```python
def load_data_from_resource(resource: Union[Text], domain: Domain, exclusion_percentage: Optional[int] = None) -> List["StoryStep"]
```

Loads core training data from the specified folder.

**Arguments**:

- `resource` - Folder/File with core training data files.
- `domain` - Domain object.
- `exclusion_percentage` - Identifies the percentage of training data that
  should be excluded from the training.
  

**Returns**:

  Story steps from the training data.

#### load\_data\_from\_files

```python
def load_data_from_files(story_files: List[Text], domain: Domain, exclusion_percentage: Optional[int] = None) -> List["StoryStep"]
```

Loads core training data from the specified files.

**Arguments**:

- `story_files` - List of files with training data in it.
- `domain` - Domain object.
- `exclusion_percentage` - Identifies the percentage of training data that
  should be excluded from the training.
  

**Returns**:

  Story steps from the training data.

