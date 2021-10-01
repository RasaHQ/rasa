---
sidebar_label: rasa.shared.nlu.training_data.formats.readerwriter
title: rasa.shared.nlu.training_data.formats.readerwriter
---
## TrainingDataReader Objects

```python
class TrainingDataReader(abc.ABC)
```

Reader for NLU training data.

#### \_\_init\_\_

```python
def __init__() -> None
```

Creates reader instance.

#### read

```python
def read(filename: Union[Text, Path], **kwargs: Any) -> "TrainingData"
```

Reads TrainingData from a file.

#### reads

```python
@abc.abstractmethod
def reads(s: Text, **kwargs: Any) -> "TrainingData"
```

Reads TrainingData from a string.

## TrainingDataWriter Objects

```python
class TrainingDataWriter()
```

A class for writing training data to a file.

#### dump

```python
def dump(filename: Text, training_data: "TrainingData") -> None
```

Writes a TrainingData object to a file.

#### dumps

```python
def dumps(training_data: "TrainingData") -> Text
```

Turns TrainingData into a string.

#### prepare\_training\_examples

```python
@staticmethod
def prepare_training_examples(training_data: "TrainingData") -> OrderedDict
```

Pre-processes training data examples by removing not trainable entities.

#### generate\_list\_item

```python
@staticmethod
def generate_list_item(text: Text) -> Text
```

Generates text for a list item.

#### generate\_message

```python
@staticmethod
def generate_message(message: Dict[Text, Any]) -> Text
```

Generates text for a message object.

#### generate\_entity

```python
@staticmethod
def generate_entity(text: Text, entity: Dict[Text, Any]) -> Text
```

Generates text for an entity object.

## JsonTrainingDataReader Objects

```python
class JsonTrainingDataReader(TrainingDataReader)
```

#### reads

```python
def reads(s: Text, **kwargs: Any) -> "TrainingData"
```

Transforms string into json object and passes it on.

#### read\_from\_json

```python
def read_from_json(js: Dict[Text, Any], **kwargs: Any) -> "TrainingData"
```

Reads TrainingData from a json object.

