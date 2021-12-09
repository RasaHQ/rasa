---
sidebar_label: rasa.nlu.training_data.formats.rasa
title: rasa.nlu.training_data.formats.rasa
---
## RasaReader Objects

```python
class RasaReader(JsonTrainingDataReader)
```

#### read\_from\_json

```python
 | read_from_json(js: Dict[Text, Any], **_) -> "TrainingData"
```

Loads training data stored in the rasa NLU data format.

## RasaWriter Objects

```python
class RasaWriter(TrainingDataWriter)
```

#### dumps

```python
 | dumps(training_data: "TrainingData", **kwargs) -> Text
```

Writes Training Data to a string in json format.

