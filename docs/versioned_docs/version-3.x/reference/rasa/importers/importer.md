---
sidebar_label: rasa.importers.importer
title: rasa.importers.importer
---
## TrainingDataImporter Objects

```python
class TrainingDataImporter()
```

Common interface for different mechanisms to load training data.

#### get\_domain

```python
 | async get_domain() -> Domain
```

Retrieves the domain of the bot.

**Returns**:

  Loaded `Domain`.

#### get\_stories

```python
 | async get_stories(template_variables: Optional[Dict] = None, use_e2e: bool = False, exclusion_percentage: Optional[int] = None) -> StoryGraph
```

Retrieves the stories that should be used for training.

**Arguments**:

- `template_variables` - Values of templates that should be replaced while
  reading the story files.
- `use_e2e` - Specifies whether to parse end to end learning annotations.
- `exclusion_percentage` - Amount of training data that should be excluded.
  

**Returns**:

  `StoryGraph` containing all loaded stories.

#### get\_config

```python
 | async get_config() -> Dict
```

Retrieves the configuration that should be used for the training.

**Returns**:

  The configuration as dictionary.

#### get\_nlu\_data

```python
 | async get_nlu_data(language: Optional[Text] = "en") -> TrainingData
```

Retrieves the NLU training data that should be used for training.

**Arguments**:

- `language` - Can be used to only load training data for a certain language.
  

**Returns**:

  Loaded NLU `TrainingData`.

#### load\_from\_config

```python
 | @staticmethod
 | load_from_config(config_path: Text, domain_path: Optional[Text] = None, training_data_paths: Optional[List[Text]] = None, training_type: Optional[TrainingType] = TrainingType.BOTH) -> "TrainingDataImporter"
```

Loads a `TrainingDataImporter` instance from a configuration file.

#### load\_core\_importer\_from\_config

```python
 | @staticmethod
 | load_core_importer_from_config(config_path: Text, domain_path: Optional[Text] = None, training_data_paths: Optional[List[Text]] = None) -> "TrainingDataImporter"
```

Loads core `TrainingDataImporter` instance.

Instance loaded from configuration file will only read Core training data.

#### load\_nlu\_importer\_from\_config

```python
 | @staticmethod
 | load_nlu_importer_from_config(config_path: Text, domain_path: Optional[Text] = None, training_data_paths: Optional[List[Text]] = None) -> "TrainingDataImporter"
```

Loads nlu `TrainingDataImporter` instance.

Instance loaded from configuration file will only read NLU training data.

#### load\_from\_dict

```python
 | @staticmethod
 | load_from_dict(config: Optional[Dict], config_path: Text, domain_path: Optional[Text] = None, training_data_paths: Optional[List[Text]] = None, training_type: Optional[TrainingType] = TrainingType.BOTH) -> "TrainingDataImporter"
```

Loads a `TrainingDataImporter` instance from a dictionary.

## NluDataImporter Objects

```python
class NluDataImporter(TrainingDataImporter)
```

Importer that skips any Core-related file reading.

## CoreDataImporter Objects

```python
class CoreDataImporter(TrainingDataImporter)
```

Importer that skips any NLU related file reading.

## CombinedDataImporter Objects

```python
class CombinedDataImporter(TrainingDataImporter)
```

A `TrainingDataImporter` that combines multiple importers.
Uses multiple `TrainingDataImporter` instances
to load the data as if they were a single instance.

## RetrievalModelsDataImporter Objects

```python
class RetrievalModelsDataImporter(TrainingDataImporter)
```

A `TrainingDataImporter` that sets up the data for training retrieval models.

Synchronizes response templates between Domain and NLU
and adds retrieval intent properties from the NLU training data
back to the Domain.

#### get\_domain

```python
 | async get_domain() -> Domain
```

Merge existing domain with properties of retrieval intents in NLU data.

#### get\_nlu\_data

```python
 | async get_nlu_data(language: Optional[Text] = "en") -> TrainingData
```

Update NLU data with response templates defined in the domain

## E2EImporter Objects

```python
class E2EImporter(TrainingDataImporter)
```

Importer which
- enhances the NLU training data with actions / user messages from the stories.
- adds potential end-to-end bot messages from stories as actions to the domain

