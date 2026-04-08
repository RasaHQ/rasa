"""Tests for ClassifierTask — thorough coverage."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from rasa.builder.evaluator.tasks.base import TaskResult

CLASSIFIER_PATH = (
    "rasa.builder.copilot.message_classifier.message_classifier.MessageClassifier"
)


class TestClassifierTaskInit:
    def test_creates_classifier_instance(self):
        with patch(CLASSIFIER_PATH) as mock_cls:
            from rasa.builder.evaluator.tasks.classifier_task import ClassifierTask

            task = ClassifierTask()
            mock_cls.assert_called_once()
            assert task._classifier is mock_cls.return_value


class TestClassifierTaskRunTask:
    async def test_success(self):
        from rasa.builder.copilot.models import ResponseCategory

        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(
            return_value=SimpleNamespace(
                category=ResponseCategory.COPILOT,
                raw_response="test response",
            )
        )

        with patch(CLASSIFIER_PATH, return_value=mock_classifier):
            from rasa.builder.evaluator.tasks.classifier_task import ClassifierTask

            task = ClassifierTask()

        item = SimpleNamespace(
            id="item-1",
            input={"message": "hello"},
            expected_output={
                "answer": "hi",
                "response_category": "copilot",
                "references": [],
            },
            metadata={
                "ids": {},
                "copilot_additional_context": {
                    "relevant_documents": [],
                    "relevant_assistant_files": {},
                    "assistant_tracker_context": None,
                    "assistant_logs": "",
                    "copilot_chat_history": [],
                },
            },
        )
        result = await task.run_task(item=item)

        assert isinstance(result, TaskResult)
        assert result.predicted_category == ResponseCategory.COPILOT
        assert result.complete_response == "test response"

    async def test_exception_returns_none(self):
        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(side_effect=RuntimeError("boom"))

        with patch(CLASSIFIER_PATH, return_value=mock_classifier):
            from rasa.builder.evaluator.tasks.classifier_task import ClassifierTask

            task = ClassifierTask()

        item = SimpleNamespace(
            id="item-1",
            input={"message": "hello"},
            expected_output={
                "answer": "hi",
                "response_category": "copilot",
                "references": [],
            },
            metadata={
                "ids": {},
                "copilot_additional_context": {
                    "relevant_documents": [],
                    "relevant_assistant_files": {},
                    "assistant_tracker_context": None,
                    "assistant_logs": "",
                    "copilot_chat_history": [],
                },
            },
        )
        result = await task.run_task(item=item)

        assert result is None

    async def test_bad_item_data_returns_none(self):
        with patch(CLASSIFIER_PATH):
            from rasa.builder.evaluator.tasks.classifier_task import ClassifierTask

            task = ClassifierTask()

        # Item with invalid structure → from_raw_data will raise
        item = SimpleNamespace(
            id="bad-item",
            input="not a dict",
            expected_output="not a dict",
            metadata="not a dict",
        )
        result = await task.run_task(item=item)

        assert result is None
