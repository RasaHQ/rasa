---
sidebar_label: rasa.nlu.training_data.synonyms_parser
title: rasa.nlu.training_data.synonyms_parser
---
#### add\_synonyms\_from\_entities

```python
add_synonyms_from_entities(plain_text: Text, entities: List[Dict], existing_synonyms: Dict[Text, Any]) -> None
```

Adds synonyms found in intent examples.

**Arguments**:

- `plain_text` - Plain (with removed special symbols) user utterance.
- `entities` - Entities that were extracted from the original user utterance.
- `existing_synonyms` - The dict with existing synonyms mappings that will
  be extended.

#### add\_synonym

```python
add_synonym(synonym_value: Text, synonym_name: Text, existing_synonyms: Dict[Text, Any]) -> None
```

Adds a new synonym mapping to the provided list of synonyms.

**Arguments**:

- `synonym_value` - Value of the synonym.
- `synonym_name` - Name of the synonym.
- `existing_synonyms` - Dictionary will synonym mappings that will be extended.

