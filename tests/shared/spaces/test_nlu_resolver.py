from typing import Text, Set

import pytest

from rasa.shared.spaces.domain_resolver import DomainResolver
from rasa.shared.spaces.nlu_resolver import NLUResolver


@pytest.mark.parametrize("prefix, domain_path, nlu_path, "
                         "expected_intents, expected_entities,"
                         "expected_lookup_features, expected_regex_features",
                         [
                             # Test 1
                             ("money",
                              "data/test_spaces/money/domain.yml",
                              "data/test_spaces/money/nlu.yml",
                              {"money!send_money", "money!receive_money", "greet"},
                              set(),
                              {"money!credit_card"},
                              {"money!account_number"}),
                             # Test 2
                             ("restaurant",
                              "data/test_restaurantbot/domain.yml",
                              "data/test_restaurantbot/data/nlu.yml",
                              {"restaurant!request_restaurant", "restaurant!inform"},
                              {"restaurant!seating",
                               "restaurant!cuisine", "number"},
                              set(),
                              set())
                         ])
def test_nlu_resolving(prefix: Text, domain_path: Text, nlu_path: Text,
                       expected_intents: Set[Text], expected_entities: Set[Text],
                       expected_lookup_features: Set[Text],
                       expected_regex_features: Set[Text]):
    _, domain_info = DomainResolver.load_and_resolve(domain_path, prefix)
    training_data = NLUResolver.load_and_resolve(nlu_path, prefix, domain_info)

    assert training_data.intents == expected_intents
    assert training_data.entities == expected_entities
    assert {t["name"] for t in training_data.lookup_tables} == expected_lookup_features
    assert {r["name"] for r in training_data.regex_features} == expected_regex_features
