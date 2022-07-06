---
sidebar_label: rasa.shared.nlu.training_data.loading
title: rasa.shared.nlu.training_data.loading
---
#### load\_data

```python
def load_data(resource_name: Text, language: Optional[Text] = "en") -> "TrainingData"
```

Load training data from disk.

Merges them if loaded from disk and multiple files are found.

#### guess\_format

```python
def guess_format(filename: Text) -> Text
```

Applies heuristics to guess the data format of a file.

**Arguments**:

- `filename` - file whose type should be guessed
  

**Returns**:

  Guessed file format.

