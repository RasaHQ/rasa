import rasa.shared.nlu.training_data.synonyms_parser as synonyms_parser


def test_add_synonym():

    synonym_name = "savings"
    synonym_examples = ["pink pig", "savings account"]
    expected_result = {"pink pig": synonym_name, "savings account": synonym_name}

    result = {}

    for example in synonym_examples:
        synonyms_parser.add_synonym(example, synonym_name, result)

    assert result == expected_result


def test_add_synonyms_from_entities():

    training_example = "I want to fly from Berlin to LA"

    entities = [
        {"start": 19, "end": 25, "value": "Berlin", "entity": "city", "role": "to"},
        {
            "start": 29,
            "end": 31,
            "value": "Los Angeles",
            "entity": "city",
            "role": "from",
        },
    ]

    result = {}

    synonyms_parser.add_synonyms_from_entities(training_example, entities, result)

    assert result == {"LA": "Los Angeles"}
