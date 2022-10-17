---
sidebar_label: rasa.nlu.training_data.formats.readerwriter
title: rasa.nlu.training_data.formats.readerwriter
---
## TrainingDataReader Objects

```python
class TrainingDataReader()
```

#### read

```python
 | read(filename: Union[Text, Path], **kwargs: Any) -> "TrainingData"
```

Reads TrainingData from a file.

#### reads

```python
 | reads(s: Text, **kwargs: Any) -> "TrainingData"
```

Reads TrainingData from a string.

## TrainingDataWriter Objects

```python
class TrainingDataWriter()
```

#### dump

```python
 | dump(filename: Text, training_data) -> None
```

Writes a TrainingData object in markdown format to a file.

#### dumps

```python
 | dumps(training_data: "TrainingData") -> Text
```

Turns TrainingData into a string.

#### prepare\_training\_examples

```python
 | @staticmethod
 | prepare_training_examples(training_data: "TrainingData") -> OrderedDict
```

Pre-processes training data examples by removing not trainable entities.

#### generate\_list\_item

```python
 | @staticmethod
 | generate_list_item(text: Text) -> Text
```

Generates text for a list item.

#### generate\_message

```python
 | @staticmethod
 | generate_message(message: Dict[Text, Any]) -> Text
```

Generates text for a message object.

#### generate\_entity

```python
 | @staticmethod
 | generate_entity(text: Text, entity: Dict[Text, Any]) -> Text
```

Generates text for an entity object.

## JsonTrainingDataReader Objects

```python
class JsonTrainingDataReader(TrainingDataReader)
```

#### reads

```python
 | reads(s: Text, **kwargs: Any) -> "TrainingData"
```

Transforms string into json object and passes it on.

#### read\_from\_json

```python
 | read_from_json(js: Dict[Text, Any], **kwargs: Any) -> "TrainingData"
```

Reads TrainingData from a json object.

