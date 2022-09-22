from typing import Text, Set

import pytest

from rasa.shared.spaces.domain_resolver import DomainResolver
from rasa.shared.spaces.nlu_resolver import NLUResolver


@pytest.mark.parametrize("prefix, domain_path, nlu_path, "
                         "expected_intents, expected_entities,"
                         "expected_lookup_features, expected_regex_features,"
                         "expected_responses",
                         [
                             # Test 1
                             ("money",
                              "data/test_spaces/money/domain.yml",
                              "data/test_spaces/money/nlu.yml",
                              {"money!send_money", "money!receive_money", "greet"},
                              set(),
                              {"money!credit_card"},
                              {"money!account_number"},
                              set()),
                             # Test 2
                             ("restaurant",
                              "data/test_restaurantbot/domain.yml",
                              "data/test_restaurantbot/data/nlu.yml",
                              {"restaurant!request_restaurant", "restaurant!inform"},
                              {"restaurant!seating",
                               "restaurant!cuisine", "number"},
                              set(),
                              set(),
                              set()),
                             ("carbon",
                              "data/test_selectors/domain.yml",
                              "data/test_selectors/nlu.yml",
                              {"carbon!faq"},
                              {"city"},
                              set(),
                              set(),
                              {"utter_carbon!faq/meta_inquire-ask_bot-challenge",
                               "utter_carbon!faq/ask_howdoing",
                               "utter_carbon!faq/meta_inquire_capabilities",
                               "utter_carbon!faq/oos-future_inquire-ask",
                               "utter_carbon!faq/oos_inform",
                               "utter_carbon!faq/oos_inquire",
                               "utter_carbon!faq/oos_request",
                               "utter_carbon!faq/placeholder",
                               "utter_carbon!faq/takeacut",
                               })
                         ])
def test_nlu_resolving(prefix: Text, domain_path: Text, nlu_path: Text,
                       expected_intents: Set[Text], expected_entities: Set[Text],
                       expected_lookup_features: Set[Text],
                       expected_regex_features: Set[Text],
                       expected_responses: Set[Text]):
    _, domain_info = DomainResolver.load_and_resolve(domain_path, prefix)
    training_data = NLUResolver.load_and_resolve(nlu_path, prefix, domain_info)

    assert training_data.intents == expected_intents
    assert training_data.entities == expected_entities
    assert {t["name"] for t in training_data.lookup_tables} == expected_lookup_features
    assert {r["name"] for r in training_data.regex_features} == expected_regex_features
    assert set(training_data.responses.keys()) == expected_responses
