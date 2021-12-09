---
sidebar_label: rasa.nlu.training_data.loading
title: rasa.nlu.training_data.loading
---
#### load\_data

```python
load_data(resource_name: Text, language: Optional[Text] = "en") -> "TrainingData"
```

Load training data from disk.

Merges them if loaded from disk and multiple files are found.

#### load\_data\_from\_endpoint

```python
async load_data_from_endpoint(data_endpoint: EndpointConfig, language: Optional[Text] = "en") -> "TrainingData"
```

Load training data from a URL.

#### guess\_format

```python
guess_format(filename: Text) -> Text
```

Applies heuristics to guess the data format of a file.

**Arguments**:

- `filename` - file whose type should be guessed
  

**Returns**:

  Guessed file format.

