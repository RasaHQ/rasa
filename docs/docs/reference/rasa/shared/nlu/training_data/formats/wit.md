---
sidebar_label: rasa.shared.nlu.training_data.formats.wit
title: rasa.shared.nlu.training_data.formats.wit
---
## WitReader Objects

```python
class WitReader(JsonTrainingDataReader)
```

#### read\_from\_json

```python
def read_from_json(js: Dict[Text, Any], **kwargs: Any) -> TrainingData
```

Loads training data stored in the WIT.ai data format.

