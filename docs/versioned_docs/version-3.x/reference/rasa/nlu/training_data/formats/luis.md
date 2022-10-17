---
sidebar_label: rasa.nlu.training_data.formats.luis
title: rasa.nlu.training_data.formats.luis
---
## LuisReader Objects

```python
class LuisReader(JsonTrainingDataReader)
```

#### read\_from\_json

```python
 | read_from_json(js: Dict[Text, Any], **kwargs: Any) -> "TrainingData"
```

Loads training data stored in the LUIS.ai data format.

