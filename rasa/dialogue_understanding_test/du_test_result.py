from typing import Optional

from pydantic import BaseModel

from rasa.dialogue_understanding_test.du_test_case import DialogueUnderstandingTestCase


class DialogueUnderstandingTestResult(BaseModel):
    test_case: DialogueUnderstandingTestCase
    passed: bool
    error_line: Optional[int] = None
