import textwrap

import pytest

from rasa.dialogue_understanding_test.du_test_case import DialogueUnderstandingTestCase
from rasa.dialogue_understanding_test.validation import validate_test_cases
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import FlowsList


def test_validate_test_cases() -> None:
    # Given
    test_cases = [
        DialogueUnderstandingTestCase.from_dict(
            input_test_case={
                "test_case": "test",
                "steps": [{"user": "hello"}, {"utter": "utter_test"}],
            },
            flows=FlowsList([]),
        )
    ]
    domain = textwrap.dedent(
        """
    responses:
      utter_test:
      - text: "Test response"
      """
    )
    domain = Domain.from_yaml(domain)

    # When
    validate_test_cases(test_cases, domain)


def test_validate_test_cases_passes_when_using_placeholder() -> None:
    # Given
    test_cases = [
        DialogueUnderstandingTestCase.from_dict(
            input_test_case={
                "test_case": "test",
                "steps": [{"user": "hello"}, {"utter": "placeholder_generated_answer"}],
            },
            flows=FlowsList([]),
        )
    ]

    # When
    validate_test_cases(test_cases, Domain.empty())


def test_validate_test_cases_fails_with_using_invalid_response_template() -> None:
    # Given
    test_cases = [
        DialogueUnderstandingTestCase.from_dict(
            input_test_case={
                "test_case": "test",
                "steps": [{"user": "hello"}, {"utter": "utter_test"}],
            },
            flows=FlowsList([]),
        )
    ]

    with pytest.raises(SystemExit):
        validate_test_cases(test_cases, Domain.empty())


def test_validate_test_cases_with_no_domain_throws_error():
    with pytest.raises(SystemExit):
        validate_test_cases([], None)
