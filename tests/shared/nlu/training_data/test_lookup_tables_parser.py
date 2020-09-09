import rasa.shared.nlu.training_data.lookup_tables_parser as lookup_tables_parser


def test_add_item_to_lookup_tables():
    lookup_item_title = "additional_currencies"
    lookup_examples = ["Peso", "Euro", "Dollar"]

    result = []

    for example in lookup_examples:
        lookup_tables_parser.add_item_to_lookup_tables(
            lookup_item_title, example, result
        )

    assert result == [{"name": lookup_item_title, "elements": lookup_examples}]
