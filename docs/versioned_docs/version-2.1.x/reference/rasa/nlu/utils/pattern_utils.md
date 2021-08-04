---
sidebar_label: pattern_utils
title: rasa.nlu.utils.pattern_utils
---

#### read\_lookup\_table\_file

```python
read_lookup_table_file(lookup_table_file: Text) -> List[Text]
```

Read the lookup table file.

**Arguments**:

- `lookup_table_file` - the file path to the lookup table
  

**Returns**:

  Elements listed in the lookup table file.

#### extract\_patterns

```python
extract_patterns(training_data: TrainingData, use_lookup_tables: bool = True, use_regexes: bool = True, use_only_entities: bool = False) -> List[Dict[Text, Text]]
```

Extract a list of patterns from the training data.

The patterns are constructed using the regex features and lookup tables defined
in the training data.

**Arguments**:

- `training_data` - The training data.
- `use_only_entities` - If True only lookup tables and regex features with a name
  equal to a entity are considered.
- `use_regexes` - Boolean indicating whether to use regex features or not.
- `use_lookup_tables` - Boolean indicating whether to use lookup tables or not.
  

**Returns**:

  The list of regex patterns.

