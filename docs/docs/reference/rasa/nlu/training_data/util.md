---
sidebar_label: rasa.nlu.training_data.util
title: rasa.nlu.training_data.util
---
#### transform\_entity\_synonyms

```python
transform_entity_synonyms(synonyms, known_synonyms: Optional[Dict[Text, Any]] = None) -> Dict[Text, Any]
```

Transforms the entity synonyms into a text-&gt;value dictionary

#### remove\_untrainable\_entities\_from

```python
remove_untrainable_entities_from(example: Dict[Text, Any]) -> None
```

Remove untrainable entities from serialised training example `example`.

Entities with an untrainable extractor will be removed. Untrainable extractors
are defined in `rasa.nlu.constants.PRETRAINED_EXTRACTORS`.

**Arguments**:

- `example` - Serialised training example to inspect.

