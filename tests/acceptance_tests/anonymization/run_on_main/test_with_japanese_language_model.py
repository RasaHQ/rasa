from typing import Text

import pytest

from rasa.anonymization.anonymization_rule_executor import (
    AnonymizationRule,
    AnonymizationRuleExecutor,
    AnonymizationRuleList,
)

# The individual tests in this `tests/integration/anonymization` package were separated
# into different modules because run together in the same pytest session would cause
# the majority to fail apart from the first parameterized test. The first test would
# load a particular language model and the subsequent tests would fail because the
# `AnonymizationAnalyzer` component is a Singleton and would not be able to load a
# different language model again.


@pytest.mark.parametrize(
    "entity_type, text",
    [
        ("PERSON", "私の名前はジェーンです"),
        ("PHONE_NUMBER", "私の電話番号は 020 123 4567 です"),
        ("EMAIL_ADDRESS", "test@test.com にメールを送ってください"),
        ("IBAN_CODE", "IBAN ES79 2100 0813 6101 2345 6789 に送金できます"),
        (
            "CREDIT_CARD",
            "このクレジット カード番号 4916741327614057 を使用してください",
        ),
        ("DATE_TIME", "私の誕生日は01/01/1990です"),
        ("IP_ADDRESS", "127.0.0.1 を調べる"),
        ("URL", "www.test.com で情報を探す必要があります"),
        ("LOCATION", "あなたはすぐに東京に行くべきです"),
    ],
)
def test_anonymization_rule_executor_in_non_default_japanese_language(
    entity_type: Text,
    text: Text,
) -> None:
    rule_list = AnonymizationRuleList(
        id="test",
        rule_list=[
            AnonymizationRule(
                entity_name=entity_type,
                substitution="mask",
            ),
        ],
        language="ja",
        models="ja_core_news_lg",
    )

    anonymization_rule_executor = AnonymizationRuleExecutor(rule_list)
    anonymized_text = anonymization_rule_executor.run(text)

    assert anonymized_text != text
