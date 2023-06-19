---
sidebar_label: rasa.shared.nlu.training_data.formats.rasa
title: rasa.shared.nlu.training_data.formats.rasa
---
## RasaReader Objects

```python
class RasaReader(JsonTrainingDataReader)
```

Reader for Rasa NLU training data in JSON format.

**Example**:

  {
- `"rasa_nlu_data"` - {
- `"regex_features"` - [
  {
- `"name"` - &quot;zipcode&quot;,
- `"pattern"` - &quot;[0-9]{5}&quot;
  }
  ],
- `"entity_synonyms"` - [
  {
- `"value"` - &quot;chinese&quot;,
- `"synonyms"` - [&quot;Chinese&quot;, &quot;Chines&quot;, &quot;chines&quot;]
  }
  ],
- `"common_examples"` - [
  {
- `"text"` - &quot;hey&quot;,
- `"intent"` - &quot;greet&quot;,
- `"regex_features"`0 - []
  },
  {
- `"text"` - &quot;howdy&quot;,
- `"intent"` - &quot;greet&quot;,
- `"regex_features"`0 - []
  }
  ]
  }
  }

#### \_\_init\_\_

```python
def __init__() -> None
```

Creates reader.

#### read\_from\_json

```python
def read_from_json(js: Dict[Text, Any], **_: Any) -> "TrainingData"
```

Loads training data stored in the rasa NLU data format.

## RasaWriter Objects

```python
class RasaWriter(TrainingDataWriter)
```

Dumps NLU data as Rasa JSON string.

#### \_\_init\_\_

```python
def __init__() -> None
```

Creates writer.

#### dumps

```python
def dumps(training_data: "TrainingData", **kwargs: Any) -> Text
```

Writes Training Data to a string in json format.

