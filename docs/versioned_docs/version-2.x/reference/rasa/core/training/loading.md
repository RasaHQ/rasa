---
sidebar_label: rasa.core.training.loading
title: rasa.core.training.loading
---
#### load\_data\_from\_resource

```python
async load_data_from_resource(resource: Union[Text, Path], domain: Domain, template_variables: Optional[Dict] = None, use_e2e: bool = False, exclusion_percentage: Optional[int] = None) -> List["StoryStep"]
```

Loads core training data from the specified folder.

**Arguments**:

- `resource` - Folder/File with core training data files.
- `domain` - Domain object.
- `template_variables` - Variables that have to be replaced in the training data.
- `use_e2e` - Identifies if the e2e reader should be used.
- `exclusion_percentage` - Identifies the percentage of training data that
  should be excluded from the training.
  

**Returns**:

  Story steps from the training data.

#### load\_data\_from\_files

```python
async load_data_from_files(story_files: List[Text], domain: Domain, template_variables: Optional[Dict] = None, use_e2e: bool = False, exclusion_percentage: Optional[int] = None) -> List["StoryStep"]
```

Loads core training data from the specified files.

**Arguments**:

- `story_files` - List of files with training data in it.
- `domain` - Domain object.
- `template_variables` - Variables that have to be replaced in the training data.
- `use_e2e` - Identifies whether the e2e reader should be used.
- `exclusion_percentage` - Identifies the percentage of training data that
  should be excluded from the training.
  

**Returns**:

  Story steps from the training data.

