---
sidebar_label: rasa.shared.nlu.training_data.lookup_tables_parser
title: rasa.shared.nlu.training_data.lookup_tables_parser
---
#### add\_item\_to\_lookup\_tables

```python
def add_item_to_lookup_tables(title: Text, item: Text, existing_lookup_tables: List[Dict[Text, Union[Text, List[Text]]]]) -> None
```

Add an item to a list of existing lookup tables.

Takes a list of lookup table dictionaries. Finds the one associated
with the current lookup, then adds the item to the list.

**Arguments**:

- `title` - Name of the lookup item.
- `item` - The lookup item.
- `existing_lookup_tables` - Existing lookup items that will be extended.
  

**Raises**:

- `TypeError` - in case we&#x27;re trying to add a lookup table element to a file.
  This is an internal error that is indicative of a parsing error.

