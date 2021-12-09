---
sidebar_label: rasa.importers.multi_project
title: rasa.importers.multi_project
---
## MultiProjectImporter Objects

```python
class MultiProjectImporter(TrainingDataImporter)
```

#### training\_paths

```python
 | training_paths() -> Set[Text]
```

Returns the paths which should be searched for training data.

#### is\_imported

```python
 | is_imported(path: Text) -> bool
```

Checks whether a path is imported by a skill.

**Arguments**:

- `path` - File or directory path which should be checked.
  

**Returns**:

  `True` if path is imported by a skill, `False` if not.

