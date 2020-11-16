---
sidebar_label: model
title: rasa.nlu.model
---

## InvalidModelError Objects

```python
class InvalidModelError(RasaException)
```

Raised when a model failed to load.

**Attributes**:

- `message` - explanation of why the model is invalid

## UnsupportedModelError Objects

```python
class UnsupportedModelError(RasaException)
```

Raised when a model is too old to be loaded.

**Attributes**:

- `message` - explanation of why the model is invalid

## Metadata Objects

```python
class Metadata()
```

Captures all information about a model to load and prepare it.

#### load

```python
 | @staticmethod
 | load(model_dir: Text)
```

Loads the metadata from a models directory.

**Arguments**:

- `model_dir` - the directory where the model is saved.

**Returns**:

- `Metadata` - A metadata object describing the model

#### language

```python
 | @property
 | language() -> Optional[Text]
```

Language of the underlying model

#### persist

```python
 | persist(model_dir: Text)
```

Persists the metadata of a model to a given directory.

## Trainer Objects

```python
class Trainer()
```

Trainer will load the data and train all components.

Requires a pipeline specification and configuration to use for
the training.

#### train

```python
 | train(data: TrainingData, **kwargs: Any) -> "Interpreter"
```

Trains the underlying pipeline using the provided training data.

#### persist

```python
 | persist(path: Text, persistor: Optional[Persistor] = None, fixed_model_name: Text = None, persist_nlu_training_data: bool = False) -> Text
```

Persist all components of the pipeline to the passed path.

Returns the directory of the persisted model.

## Interpreter Objects

```python
class Interpreter()
```

Use a trained pipeline of components to parse text messages.

#### load

```python
 | @staticmethod
 | load(model_dir: Text, component_builder: Optional[ComponentBuilder] = None, skip_validation: bool = False) -> "Interpreter"
```

Create an interpreter based on a persisted model.

**Arguments**:

- `skip_validation` - If set to `True`, does not check that all
  required packages for the components are installed
  before loading them.
- `model_dir` - The path of the model to load
- `component_builder` - The
  :class:`rasa.nlu.components.ComponentBuilder` to use.
  

**Returns**:

  An interpreter that uses the loaded model.

#### create

```python
 | @staticmethod
 | create(model_metadata: Metadata, component_builder: Optional[ComponentBuilder] = None, skip_validation: bool = False) -> "Interpreter"
```

Load stored model and components defined by the provided metadata.

#### parse

```python
 | parse(text: Text, time: Optional[datetime.datetime] = None, only_output_properties: bool = True) -> Dict[Text, Any]
```

Parse the input text, classify it and return pipeline result.

The pipeline result usually contains intent and entities.

#### featurize\_message

```python
 | featurize_message(message: Message) -> Message
```

Tokenize and featurize the input message

**Arguments**:

- `message` - message storing text to process;

**Returns**:

- `message` - it contains the tokens and features which are the output of the NLU pipeline;

