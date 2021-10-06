---
sidebar_label: rasa.nlu.model
title: rasa.nlu.model
---
## InvalidModelError Objects

```python
class InvalidModelError(RasaException)
```

Raised when a model failed to load.

**Attributes**:

- `message` - explanation of why the model is invalid

#### \_\_init\_\_

```python
 | __init__(message: Text) -> None
```

Initialize message attribute.

## UnsupportedModelError Objects

```python
class UnsupportedModelError(RasaException)
```

Raised when a model is too old to be loaded.

**Attributes**:

- `message` - explanation of why the model is invalid

#### \_\_init\_\_

```python
 | __init__(message: Text) -> None
```

Initialize message attribute.

## Metadata Objects

```python
class Metadata()
```

Captures all information about a model to load and prepare it.

#### load

```python
 | @staticmethod
 | load(model_dir: Text) -> "Metadata"
```

Loads the metadata from a models directory.

**Arguments**:

- `model_dir` - the directory where the model is saved.

**Returns**:

- `Metadata` - A metadata object describing the model

#### \_\_init\_\_

```python
 | __init__(metadata: Dict[Text, Any]) -> None
```

Set `metadata` attribute.

#### get

```python
 | get(property_name: Text, default: Any = None) -> Any
```

Proxy function to get property on `metadata` attribute.

#### component\_classes

```python
 | @property
 | component_classes() -> List[Optional[Text]]
```

Returns a list of component class names.

#### number\_of\_components

```python
 | @property
 | number_of_components() -> int
```

Returns count of components.

#### for\_component

```python
 | for_component(index: int, defaults: Any = None) -> Dict[Text, Any]
```

Returns the configuration of the component based on index.

#### language

```python
 | @property
 | language() -> Optional[Text]
```

Language of the underlying model

#### persist

```python
 | persist(model_dir: Text) -> None
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
 | load(model_dir: Text, component_builder: Optional[ComponentBuilder] = None, skip_validation: bool = False, new_config: Optional[Dict] = None, finetuning_epoch_fraction: float = 1.0) -> "Interpreter"
```

Create an interpreter based on a persisted model.

**Arguments**:

- `skip_validation` - If set to `True`, does not check that all
  required packages for the components are installed
  before loading them.
- `model_dir` - The path of the model to load
- `component_builder` - The
  :class:`rasa.nlu.components.ComponentBuilder` to use.
- `new_config` - Optional new config to use for the new epochs.
- `finetuning_epoch_fraction` - Value to multiply all epochs by.
  

**Returns**:

  An interpreter that uses the loaded model.

#### create

```python
 | @staticmethod
 | create(model_dir: Text, model_metadata: Metadata, component_builder: Optional[ComponentBuilder] = None, skip_validation: bool = False, should_finetune: bool = False) -> "Interpreter"
```

Create model and components defined by the provided metadata.

**Arguments**:

- `model_dir` - The directory containing the model.
- `model_metadata` - The metadata describing each component.
- `component_builder` - The
  :class:`rasa.nlu.components.ComponentBuilder` to use.
- `skip_validation` - If set to `True`, does not check that all
  required packages for the components are installed
  before loading them.
- `should_finetune` - Indicates if the model components will be fine-tuned.
  

**Returns**:

  An interpreter that uses the created model.

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

- `message` - it contains the tokens and features which are the output of the
  NLU pipeline;

#### warn\_of\_overlapping\_entities

```python
 | warn_of_overlapping_entities(message: Message) -> None
```

Issues a warning when there are overlapping entity annotations.

This warning is only issued once per Interpreter life time.

**Arguments**:

- `message` - user message with all processing metadata such as entities

