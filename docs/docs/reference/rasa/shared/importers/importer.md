---
sidebar_label: rasa.shared.importers.importer
title: rasa.shared.importers.importer
---
## TrainingDataImporter Objects

```python
class TrainingDataImporter()
```

Common interface for different mechanisms to load training data.

#### get\_domain

```python
def get_domain() -> Domain
```

Retrieves the domain of the bot.

**Returns**:

  Loaded `Domain`.

#### get\_stories

```python
def get_stories(exclusion_percentage: Optional[int] = None) -> StoryGraph
```

Retrieves the stories that should be used for training.

**Arguments**:

- `exclusion_percentage` - Amount of training data that should be excluded.
  

**Returns**:

  `StoryGraph` containing all loaded stories.

#### get\_conversation\_tests

```python
def get_conversation_tests() -> StoryGraph
```

Retrieves end-to-end conversation stories for testing.

**Returns**:

  `StoryGraph` containing all loaded stories.

#### get\_config

```python
def get_config() -> Dict
```

Retrieves the configuration that should be used for the training.

**Returns**:

  The configuration as dictionary.

#### get\_nlu\_data

```python
def get_nlu_data(language: Optional[Text] = "en") -> TrainingData
```

Retrieves the NLU training data that should be used for training.

**Arguments**:

- `language` - Can be used to only load training data for a certain language.
  

**Returns**:

  Loaded NLU `TrainingData`.

#### load\_from\_config

```python
@staticmethod
def load_from_config(config_path: Text, domain_path: Optional[Text] = None, training_data_paths: Optional[List[Text]] = None, training_type: Optional[TrainingType] = TrainingType.BOTH) -> "TrainingDataImporter"
```

Loads a `TrainingDataImporter` instance from a configuration file.

#### load\_core\_importer\_from\_config

```python
@staticmethod
def load_core_importer_from_config(config_path: Text, domain_path: Optional[Text] = None, training_data_paths: Optional[List[Text]] = None) -> "TrainingDataImporter"
```

Loads core `TrainingDataImporter` instance.

Instance loaded from configuration file will only read Core training data.

#### load\_nlu\_importer\_from\_config

```python
@staticmethod
def load_nlu_importer_from_config(config_path: Text, domain_path: Optional[Text] = None, training_data_paths: Optional[List[Text]] = None) -> "TrainingDataImporter"
```

Loads nlu `TrainingDataImporter` instance.

Instance loaded from configuration file will only read NLU training data.

#### load\_from\_dict

```python
@staticmethod
def load_from_dict(config: Optional[Dict] = None, config_path: Optional[Text] = None, domain_path: Optional[Text] = None, training_data_paths: Optional[List[Text]] = None, training_type: Optional[TrainingType] = TrainingType.BOTH) -> "TrainingDataImporter"
```

Loads a `TrainingDataImporter` instance from a dictionary.

#### fingerprint

```python
def fingerprint() -> Text
```

Returns a random fingerprint as data shouldn&#x27;t be cached.

#### \_\_repr\_\_

```python
def __repr__() -> Text
```

Returns text representation of object.

## NluDataImporter Objects

```python
class NluDataImporter(TrainingDataImporter)
```

Importer that skips any Core-related file reading.

#### \_\_init\_\_

```python
def __init__(actual_importer: TrainingDataImporter)
```

Initializes the NLUDataImporter.

#### get\_domain

```python
def get_domain() -> Domain
```

Retrieves model domain (see parent class for full docstring).

#### get\_stories

```python
def get_stories(exclusion_percentage: Optional[int] = None) -> StoryGraph
```

Retrieves training stories / rules (see parent class for full docstring).

#### get\_conversation\_tests

```python
def get_conversation_tests() -> StoryGraph
```

Retrieves conversation test stories (see parent class for full docstring).

#### get\_config

```python
def get_config() -> Dict
```

Retrieves model config (see parent class for full docstring).

#### get\_nlu\_data

```python
def get_nlu_data(language: Optional[Text] = "en") -> TrainingData
```

Retrieves NLU training data (see parent class for full docstring).

## CombinedDataImporter Objects

```python
class CombinedDataImporter(TrainingDataImporter)
```

A `TrainingDataImporter` that combines multiple importers.

Uses multiple `TrainingDataImporter` instances
to load the data as if they were a single instance.

#### get\_config

```python
@rasa.shared.utils.common.cached_method
def get_config() -> Dict
```

Retrieves model config (see parent class for full docstring).

#### get\_domain

```python
@rasa.shared.utils.common.cached_method
def get_domain() -> Domain
```

Retrieves model domain (see parent class for full docstring).

#### get\_stories

```python
@rasa.shared.utils.common.cached_method
def get_stories(exclusion_percentage: Optional[int] = None) -> StoryGraph
```

Retrieves training stories / rules (see parent class for full docstring).

#### get\_conversation\_tests

```python
@rasa.shared.utils.common.cached_method
def get_conversation_tests() -> StoryGraph
```

Retrieves conversation test stories (see parent class for full docstring).

#### get\_nlu\_data

```python
@rasa.shared.utils.common.cached_method
def get_nlu_data(language: Optional[Text] = "en") -> TrainingData
```

Retrieves NLU training data (see parent class for full docstring).

## ResponsesSyncImporter Objects

```python
class ResponsesSyncImporter(TrainingDataImporter)
```

Importer that syncs `responses` between Domain and NLU training data.

Synchronizes responses between Domain and NLU and
adds retrieval intent properties from the NLU training data
back to the Domain.

#### \_\_init\_\_

```python
def __init__(importer: TrainingDataImporter)
```

Initializes the ResponsesSyncImporter.

#### get\_config

```python
def get_config() -> Dict
```

Retrieves model config (see parent class for full docstring).

#### get\_domain

```python
@rasa.shared.utils.common.cached_method
def get_domain() -> Domain
```

Merge existing domain with properties of retrieval intents in NLU data.

#### get\_stories

```python
def get_stories(exclusion_percentage: Optional[int] = None) -> StoryGraph
```

Retrieves training stories / rules (see parent class for full docstring).

#### get\_conversation\_tests

```python
def get_conversation_tests() -> StoryGraph
```

Retrieves conversation test stories (see parent class for full docstring).

#### get\_nlu\_data

```python
@rasa.shared.utils.common.cached_method
def get_nlu_data(language: Optional[Text] = "en") -> TrainingData
```

Updates NLU data with responses for retrieval intents from domain.

## E2EImporter Objects

```python
class E2EImporter(TrainingDataImporter)
```

Importer with the following functionality.

- enhances the NLU training data with actions / user messages from the stories.
- adds potential end-to-end bot messages from stories as actions to the domain

#### \_\_init\_\_

```python
def __init__(importer: TrainingDataImporter) -> None
```

Initializes the E2EImporter.

#### get\_domain

```python
@rasa.shared.utils.common.cached_method
def get_domain() -> Domain
```

Retrieves model domain (see parent class for full docstring).

#### get\_stories

```python
def get_stories(exclusion_percentage: Optional[int] = None) -> StoryGraph
```

Retrieves the stories that should be used for training.

See parent class for details.

#### get\_conversation\_tests

```python
def get_conversation_tests() -> StoryGraph
```

Retrieves conversation test stories (see parent class for full docstring).

#### get\_config

```python
def get_config() -> Dict
```

Retrieves model config (see parent class for full docstring).

#### get\_nlu\_data

```python
@rasa.shared.utils.common.cached_method
def get_nlu_data(language: Optional[Text] = "en") -> TrainingData
```

Retrieves NLU training data (see parent class for full docstring).

