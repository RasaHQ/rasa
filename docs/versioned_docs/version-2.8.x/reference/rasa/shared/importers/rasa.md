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
 | async get_config() -> Dict
```

Retrieves model config (see parent class for full docstring).

#### get\_stories

```python
 | async get_stories(template_variables: Optional[Dict] = None, use_e2e: bool = False, exclusion_percentage: Optional[int] = None) -> StoryGraph
```

Retrieves training stories / rules (see parent class for full docstring).

#### get\_conversation\_tests

```python
 | async get_conversation_tests() -> StoryGraph
```

Retrieves conversation test stories (see parent class for full docstring).

#### get\_nlu\_data

```python
 | async get_nlu_data(language: Optional[Text] = "en") -> TrainingData
```

Retrieves NLU training data (see parent class for full docstring).

#### get\_domain

```python
 | async get_domain() -> Domain
```

Retrieves model domain (see parent class for full docstring).

