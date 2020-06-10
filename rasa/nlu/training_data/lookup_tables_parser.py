from typing import Text, List


class LookupTablesParser:
    @staticmethod
    def add_item_to_lookup_tables(
        title: Text, item: Text, existing_lookup_tables: List
    ) -> None:
        """Takes a list of lookup table dictionaries.  Finds the one associated
        with the current lookup, then adds the item to the list."""
        matches = [table for table in existing_lookup_tables if table["name"] == title]
        if not matches:
            existing_lookup_tables.append({"name": title, "elements": [item]})
        else:
            elements = matches[0]["elements"]
            elements.append(item)
