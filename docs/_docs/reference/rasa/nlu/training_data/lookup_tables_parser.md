---
sidebar_label: rasa.nlu.training_data.lookup_tables_parser
title: rasa.nlu.training_data.lookup_tables_parser
---
#### add\_item\_to\_lookup\_tables

```python
add_item_to_lookup_tables(title: Text, item: Text, existing_lookup_tables: List[Dict[Text, List[Text]]]) -> None
```

Takes a list of lookup table dictionaries.  Finds the one associated
with the current lookup, then adds the item to the list.

**Arguments**:

- `title` - Name of the lookup item.
- `item` - The lookup item.
- `existing_lookup_tables` - Existing lookup items that will be extended.

