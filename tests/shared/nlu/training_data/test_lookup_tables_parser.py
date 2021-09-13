import pytest

import rasa.shared.nlu.training_data.lookup_tables_parser as lookup_tables_parser


def test_add_item_to_lookup_tables():
    lookup_item_title = "additional_currencies"
    lookup_examples = ["Peso", "Euro", "Dollar"]

    lookup_tables = []

    for example in lookup_examples:
        lookup_tables_parser.add_item_to_lookup_tables(
            lookup_item_title, example, lookup_tables
        )

    assert lookup_tables == [{"name": lookup_item_title, "elements": lookup_examples}]


def test_add_item_to_lookup_tables_unloaded_file():
    lookup_item_title = "additional_currencies"

    lookup_tables = [{"name": lookup_item_title, "elements": "lookup.txt"}]

    with pytest.raises(TypeError):
        lookup_tables_parser.add_item_to_lookup_tables(
            lookup_item_title, "Pound", lookup_tables
        )
