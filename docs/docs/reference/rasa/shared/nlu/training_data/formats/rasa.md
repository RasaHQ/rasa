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
- `&quot;rasa_nlu_data&quot;` - {
- `&quot;regex_features&quot;` - [
  {
- `&quot;name&quot;` - &quot;zipcode&quot;,
- `&quot;pattern&quot;` - &quot;[0-9]{5}&quot;
  }
  ],
- `&quot;entity_synonyms&quot;` - [
  {
- `&quot;value&quot;` - &quot;chinese&quot;,
- `&quot;synonyms&quot;` - [&quot;Chinese&quot;, &quot;Chines&quot;, &quot;chines&quot;]
  }
  ],
- `&quot;common_examples&quot;` - [
  {
- `&quot;text&quot;` - &quot;hey&quot;,
- `&quot;intent&quot;` - &quot;greet&quot;,
- `&quot;entities&quot;` - []
  },
  {
- `&quot;text&quot;` - &quot;howdy&quot;,
- `&quot;intent&quot;` - &quot;greet&quot;,
- `&quot;entities&quot;` - []
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

