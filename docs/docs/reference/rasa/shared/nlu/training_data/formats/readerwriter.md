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
 | __init__() -> None
```

Creates reader instance.

#### read

```python
 | read(filename: Union[Text, Path], **kwargs: Any) -> "TrainingData"
```

Reads TrainingData from a file.

#### reads

```python
 | @abc.abstractmethod
 | reads(s: Text, **kwargs: Any) -> "TrainingData"
```

Reads TrainingData from a string.

## TrainingDataWriter Objects

```python
class TrainingDataWriter()
```

A class for writing training data to a file.

#### dump

```python
 | dump(filename: Text, training_data: "TrainingData") -> None
```

Writes a TrainingData object to a file.

#### dumps

```python
 | dumps(training_data: "TrainingData") -> Text
```

Turns TrainingData into a string.

#### prepare\_training\_examples

```python
 | @staticmethod
 | prepare_training_examples(training_data: "TrainingData") -> Dict[Text, List[Union[Dict, Text]]]
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

**Arguments**:

- `message` - A message
  

**Returns**:

  The text of the message, annotated with the entity data that is contained
  in the message

#### generate\_entity\_attributes

```python
 | @staticmethod
 | generate_entity_attributes(text: Text, entity: Dict[Text, Any], short_allowed: bool = True) -> Text
```

Generates text for the entity attributes.

**Arguments**:

- `text` - The text that is annotated with the entity
- `entity` - Entity data
- `short_allowed` - If `True`, allow shorthand annotation with parenthesis
  

**Returns**:

  The annotation text that should follow the given text

#### generate\_entity

```python
 | @staticmethod
 | generate_entity(text: Text, entity: Union[Dict[Text, Any], List[Dict[Text, Any]]]) -> Text
```

Generates text for one or multiple entity objects.

**Arguments**:

- `text` - The un-annotated text
- `entity` - One or multiple entity annotations for one part of this text
  

**Returns**:

  Annotated part of the text

## JsonTrainingDataReader Objects

```python
class JsonTrainingDataReader(TrainingDataReader)
```

A class for reading JSON files.

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

