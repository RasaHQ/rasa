---
sidebar_label: rasa.shared.importers.rasa
title: rasa.shared.importers.rasa
---
## RasaFileImporter Objects

```python
class RasaFileImporter(TrainingDataImporter)
```

Default `TrainingFileImporter` implementation.

#### get\_config

```python
 | get_config() -> Dict
```

Retrieves model config (see parent class for full docstring).

#### get\_config\_file\_for\_auto\_config

```python
 | @rasa.shared.utils.common.cached_method
 | get_config_file_for_auto_config() -> Optional[Text]
```

Returns config file path for auto-config only if there is a single one.

#### get\_stories

```python
 | get_stories(exclusion_percentage: Optional[int] = None) -> StoryGraph
```

Retrieves training stories / rules (see parent class for full docstring).

#### get\_conversation\_tests

```python
 | get_conversation_tests() -> StoryGraph
```

Retrieves conversation test stories (see parent class for full docstring).

#### get\_nlu\_data

```python
 | get_nlu_data(language: Optional[Text] = "en") -> TrainingData
```

Retrieves NLU training data (see parent class for full docstring).

#### get\_domain

```python
 | get_domain() -> Domain
```

Retrieves model domain (see parent class for full docstring).

