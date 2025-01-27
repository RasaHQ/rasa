from typing import Dict, List

import pytest

from rasa.dialogue_understanding_test.constants import ACTOR_USER
from rasa.dialogue_understanding_test.du_test_case import (
    DialogueUnderstandingTestCase,
    DialogueUnderstandingTestStep,
)
from rasa.dialogue_understanding_test.utils import (
    filter_metadata,
)
from rasa.e2e_test.e2e_test_case import Metadata


@pytest.mark.parametrize(
    "test_case, user_step, metadata, expected",
    [
        (
            DialogueUnderstandingTestCase(
                name="test_case",
                steps=[
                    DialogueUnderstandingTestStep(actor=ACTOR_USER, text="dummy step")
                ],
                metadata_name="test_case_meta",
            ),
            DialogueUnderstandingTestStep(
                actor=ACTOR_USER, metadata_name="step_meta", text="hello"
            ),
            [
                Metadata(name="test_case_meta", metadata={"key1": "value1"}),
                Metadata(name="step_meta", metadata={"key2": "value2"}),
            ],
            {"key1": "value1", "key2": "value2"},
        ),
        (
            DialogueUnderstandingTestCase(
                name="test_case",
                steps=[
                    DialogueUnderstandingTestStep(actor=ACTOR_USER, text="dummy step")
                ],
                metadata_name="test_case_meta",
            ),
            DialogueUnderstandingTestStep(
                actor=ACTOR_USER, metadata_name="step_meta", text="hello"
            ),
            [Metadata(name="test_case_meta", metadata={"key1": "value1"})],
            {"key1": "value1"},
        ),
        (
            DialogueUnderstandingTestCase(
                name="test_case",
                steps=[
                    DialogueUnderstandingTestStep(actor=ACTOR_USER, text="dummy step")
                ],
                metadata_name="test_case_meta",
            ),
            DialogueUnderstandingTestStep(
                actor=ACTOR_USER, metadata_name="step_meta", text="hello"
            ),
            [],
            {},
        ),
    ],
)
def test_filter_metadata(
    test_case: DialogueUnderstandingTestCase,
    user_step: DialogueUnderstandingTestStep,
    metadata: List[Metadata],
    expected: Dict[str, str],
):
    result = filter_metadata(test_case, user_step, metadata, "sender_id")
    assert result == expected
