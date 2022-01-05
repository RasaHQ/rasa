---
sidebar_label: rasa.validator
title: rasa.validator
---
## Validator Objects

```python
class Validator()
```

A class used to verify usage of intents and utterances.

#### \_\_init\_\_

```python
 | __init__(domain: Domain, intents: TrainingData, story_graph: StoryGraph, config: Optional[Dict[Text, Any]]) -> None
```

Initializes the Validator object.

**Arguments**:

- `domain` - The domain.
- `intents` - Training data.
- `story_graph` - The story graph.
- `config` - The configuration.

#### from\_importer

```python
 | @classmethod
 | async from_importer(cls, importer: TrainingDataImporter) -> "Validator"
```

Create an instance from the domain, nlu and story files.

#### verify\_intents

```python
 | verify_intents(ignore_warnings: bool = True) -> bool
```

Compares list of intents in domain with intents in NLU training data.

#### verify\_example\_repetition\_in\_intents

```python
 | verify_example_repetition_in_intents(ignore_warnings: bool = True) -> bool
```

Checks if there is no duplicated example in different intents.

#### verify\_intents\_in\_stories

```python
 | verify_intents_in_stories(ignore_warnings: bool = True) -> bool
```

Checks intents used in stories.

Verifies if the intents used in the stories are valid, and whether
all valid intents are used in the stories.

#### verify\_utterances\_in\_stories

```python
 | verify_utterances_in_stories(ignore_warnings: bool = True) -> bool
```

Verifies usage of utterances in stories.

Checks whether utterances used in the stories are valid,
and whether all valid utterances are used in stories.

#### verify\_actions\_in\_stories\_rules

```python
 | verify_actions_in_stories_rules() -> bool
```

Verifies that actions used in stories and rules are present in the domain.

#### verify\_story\_structure

```python
 | verify_story_structure(ignore_warnings: bool = True, max_history: Optional[int] = None) -> bool
```

Verifies that the bot behaviour in stories is deterministic.

**Arguments**:

- `ignore_warnings` - When `True`, return `True` even if conflicts were found.
- `max_history` - Maximal number of events to take into account for conflict
  identification.
  

**Returns**:

  `False` is a conflict was found and `ignore_warnings` is `False`.
  `True` otherwise.

#### verify\_nlu

```python
 | verify_nlu(ignore_warnings: bool = True) -> bool
```

Runs all the validations on intents and utterances.

#### verify\_form\_slots

```python
 | verify_form_slots() -> bool
```

Verifies that form slots match the slot mappings in domain.

#### verify\_domain\_validity

```python
 | verify_domain_validity() -> bool
```

Checks whether the domain returned by the importer is empty.

An empty domain or one that uses deprecated Mapping Policy is invalid.

#### verify\_domain\_duplicates

```python
 | verify_domain_duplicates() -> bool
```

Verifies that there are no duplicated dictionaries in multiple domain files.

**Returns**:

  `True` if duplicates exist.

