---
sidebar_label: rasa.shared.nlu.training_data.formats.luis
title: rasa.shared.nlu.training_data.formats.luis
---
## LuisReader Objects

```python
class LuisReader(JsonTrainingDataReader)
```

Reads LUIS training data.

#### read\_from\_json

```python
def read_from_json(js: Dict[Text, Any], **kwargs: Any) -> "TrainingData"
```

Loads training data stored in the LUIS.ai data format.

