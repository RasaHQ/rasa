from typing import Text, List, Dict, Union


def add_item_to_lookup_tables(
    title: Text,
    item: Text,
    existing_lookup_tables: List[Dict[Text, Union[Text, List[Text]]]],
) -> None:
    """Add an item to a list of existing lookup tables.

    Takes a list of lookup table dictionaries. Finds the one associated
    with the current lookup, then adds the item to the list.

    Args:
        title: Name of the lookup item.
        item: The lookup item.
        existing_lookup_tables: Existing lookup items that will be extended.

    Raises:
        TypeError: in case we're trying to add a lookup table element to a file.
                   This is an internal error that is indicative of a parsing error.
    """
    matches = [table for table in existing_lookup_tables if table["name"] == title]
    if not matches:
        existing_lookup_tables.append({"name": title, "elements": [item]})
    else:
        elements = matches[0]["elements"]
        if not isinstance(elements, list):
            raise TypeError("Cannot add a lookup table element to an unloaded file.")
        elements.append(item)
