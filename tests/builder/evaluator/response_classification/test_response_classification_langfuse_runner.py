"""Tests for ResponseClassificationLangfuseRunner."""

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langfuse import Evaluation
from langfuse._client.datasets import DatasetClient
from langfuse.experiment import (
    ExperimentItem,
    ExperimentItemResult,
    ExperimentResult,
)

from rasa.builder.copilot.models import (
    CopilotContext,
    ResponseCategory,
)
from rasa.builder.evaluator.copilot_executor import CopilotRunResult
from rasa.builder.evaluator.response_classification.langfuse_runner import (
    ResponseClassificationLangfuseRunner,
)


class TestResponseClassificationLangfuseRunner:
    """Test suite for ResponseClassificationLangfuseRunner."""

    @pytest.fixture
    def mock_langfuse_client(self) -> MagicMock:
        """Mock Langfuse client."""
        mock_client = MagicMock()
        mock_client._tracing_enabled = False
        return mock_client

    @pytest.fixture
    def mock_dataset(self) -> MagicMock:
        """Mock dataset client."""
        return MagicMock(spec=DatasetClient)

    @pytest.fixture
    def mock_experiment_result(self) -> MagicMock:
        """Mock experiment result."""
        result = MagicMock(spec=ExperimentResult)
        result.format.return_value = "Mock experiment result"
        return result

    @patch(
        "rasa.builder.evaluator.response_classification.langfuse_runner.langfuse.get_client"
    )
    def test_run_experiment(
        self,
        mock_get_client: MagicMock,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
        mock_experiment_result: MagicMock,
    ) -> None:
        """Test run_experiment method."""
        # Given
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset
        mock_dataset.run_experiment.return_value = mock_experiment_result

        runner = ResponseClassificationLangfuseRunner("test_dataset")

        # When
        runner.run_experiment()

        # Then
        mock_dataset.run_experiment.assert_called_once()
        call_args = mock_dataset.run_experiment.call_args
        assert call_args[1]["name"] == "Copilot Response Classification Evaluation"
        assert "description" in call_args[1]
        assert "task" in call_args[1]
        assert "run_evaluators" in call_args[1]
        mock_langfuse_client.flush.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "rasa.builder.evaluator.response_classification.langfuse_runner"
        ".run_copilot_with_response_handler"
    )
    @patch(
        "rasa.builder.evaluator.response_classification.langfuse_runner.langfuse"
        ".get_client"
    )
    async def test_run_copilot_task_success(
        self,
        mock_get_client: MagicMock,
        mock_run_copilot_with_response_handler: MagicMock,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
    ) -> None:
        """Test _run_copilot_task method with successful execution."""
        # Given

        runner = ResponseClassificationLangfuseRunner("test_dataset")
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset
        # Mock the return result of the run_copilot_with_response_handler function
        test_copilot_result = AsyncMock(spec=CopilotRunResult)
        mock_run_copilot_with_response_handler.return_value = test_copilot_result

        # Mock the experiment item that is passed to the _run_copilot_task method
        experiment_item = MagicMock(spec=ExperimentItem)
        experiment_item.id = "test_item_1"
        experiment_item.input = {"test": "input"}
        experiment_item.expected_output = {
            "answer": "Test answer",
            "response_category": "copilot",
            "references": [
                {"index": 0, "title": "test title", "url": "https://test.com"}
            ],
        }
        experiment_item.metadata = {
            "copilot_additional_context": {
                "assistant_logs": "test logs",
                "relevant_assistant_files": {"file1.py": "content1"},
                "assistant_tracker_context": None,
                "copilot_chat_history": [],
            }
        }

        # Create expected context that should be passed to
        # run_copilot_with_response_handler. It's based on the experiment item metadata.
        copilot_context = CopilotContext(
            tracker_context=None,
            assistant_logs="test logs",
            assistant_files={"file1.py": "content1"},
            copilot_chat_history=[],
        )

        # When
        result = await runner._run_copilot_task(item=experiment_item)

        # Then
        assert result == test_copilot_result
        mock_run_copilot_with_response_handler.assert_called_once_with(copilot_context)

    @pytest.mark.asyncio
    @patch(
        "rasa.builder.evaluator.response_classification.langfuse_runner"
        ".DatasetEntry.from_raw_data"
    )
    @patch(
        "rasa.builder.evaluator.response_classification.langfuse_runner"
        ".langfuse.get_client"
    )
    async def test_run_copilot_task_context_creation_failure(
        self,
        mock_get_client: MagicMock,
        mock_from_raw_data: MagicMock,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
    ) -> None:
        """Test _run_copilot_task method when context creation fails."""
        # Given
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset
        mock_from_raw_data.side_effect = Exception("Context creation failed")

        runner = ResponseClassificationLangfuseRunner("test_dataset")

        experiment_item = MagicMock(spec=ExperimentItem)
        experiment_item.id = "test_item_1"
        experiment_item.input = {"test": "input"}
        experiment_item.expected_output = {"response_category": "copilot"}
        experiment_item.metadata = {"test": "metadata"}

        # When
        result = await runner._run_copilot_task(item=experiment_item)

        # Then
        assert result is None

    @pytest.mark.asyncio
    @patch(
        "rasa.builder.evaluator.response_classification.langfuse_runner"
        ".run_copilot_with_response_handler"
    )
    @patch(
        "rasa.builder.evaluator.response_classification.langfuse_runner"
        ".DatasetEntry.from_raw_data"
    )
    @patch(
        "rasa.builder.evaluator.response_classification.langfuse_runner"
        ".langfuse.get_client"
    )
    async def test_run_copilot_task_copilot_run_failure(
        self,
        mock_get_client: MagicMock,
        mock_from_raw_data: MagicMock,
        mock_run_copilot: MagicMock,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
    ) -> None:
        """Test _run_copilot_task method when copilot run fails."""
        # Given
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset

        mock_dataset_entry = MagicMock()
        mock_copilot_context = MagicMock()
        mock_dataset_entry.to_copilot_context.return_value = mock_copilot_context
        mock_from_raw_data.return_value = mock_dataset_entry
        mock_run_copilot.side_effect = Exception("Copilot run failed")

        runner = ResponseClassificationLangfuseRunner("test_dataset")

        experiment_item = MagicMock(spec=ExperimentItem)
        experiment_item.id = "test_item_1"
        experiment_item.input = {"test": "input"}
        experiment_item.expected_output = {"response_category": "copilot"}
        experiment_item.metadata = {"test": "metadata"}

        # When
        result = await runner._run_copilot_task(item=experiment_item)

        # Then
        assert result is None

    @patch(
        "rasa.builder.evaluator.response_classification.langfuse_runner"
        ".langfuse.get_client"
    )
    def test_run_classification_metrics_evaluator(
        self,
        mock_get_client: MagicMock,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
    ) -> None:
        """Test _run_classification_metrics_evaluator with good and bad results."""
        # Given
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset

        runner = ResponseClassificationLangfuseRunner("test_dataset")

        # Create 2 good item results (valid data)
        good_item_result_1 = MagicMock(spec=ExperimentItemResult)
        good_item_result_1.output = MagicMock(spec=CopilotRunResult)
        good_item_result_1.output.response_category = ResponseCategory.COPILOT
        good_item_result_1.item = MagicMock(spec=ExperimentItem)
        good_item_result_1.item.id = "good_item_1"
        good_item_result_1.item.expected_output = {"response_category": "copilot"}

        good_item_result_2 = MagicMock(spec=ExperimentItemResult)
        good_item_result_2.output = MagicMock(spec=CopilotRunResult)
        good_item_result_2.output.response_category = (
            ResponseCategory.OUT_OF_SCOPE_DETECTION
        )
        good_item_result_2.item = MagicMock(spec=ExperimentItem)
        good_item_result_2.item.id = "good_item_2"
        good_item_result_2.item.expected_output = {
            "response_category": "out_of_scope_detection"
        }

        # Create 2 bad item results (invalid data that will be skipped)
        # Item result 1: Invalid output
        bad_item_result_1 = MagicMock(spec=ExperimentItemResult)
        bad_item_result_1.output = None  # Invalid output
        bad_item_result_1.item = MagicMock(spec=ExperimentItem)
        bad_item_result_1.item.id = "bad_item_1"
        bad_item_result_1.item.expected_output = {"response_category": "copilot"}

        # Item result 2: Invalid expected output
        bad_item_result_2 = MagicMock(spec=ExperimentItemResult)
        bad_item_result_2.output = MagicMock(spec=CopilotRunResult)
        bad_item_result_2.output.response_category = ResponseCategory.COPILOT
        bad_item_result_2.item = MagicMock(spec=ExperimentItem)
        bad_item_result_2.item.id = "bad_item_2"
        bad_item_result_2.item.expected_output = None

        item_results: List[ExperimentItemResult] = [
            good_item_result_1,
            good_item_result_2,
            bad_item_result_1,
            bad_item_result_2,
        ]

        # When
        evaluations = runner._run_classification_metrics_evaluator(
            item_results=item_results
        )

        # Then
        assert isinstance(evaluations, list)
        assert all(isinstance(eval_obj, Evaluation) for eval_obj in evaluations)
        # Verify we have evaluations from the response classification evaluator
        assert len(evaluations) > 0
        # Verify that the last evaluation is the skip count and that we have 2 skipped
        # items
        assert evaluations[-1].name == "skipped_items"
        assert evaluations[-1].value == 2

    @patch(
        "rasa.builder.evaluator.response_classification.langfuse_runner.langfuse.get_client"
    )
    def test_create_classification_results_from_dataset_items_success(
        self,
        mock_get_client: MagicMock,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
    ) -> None:
        """Test _create_classification_results_from_dataset_items with mixed data."""
        # Given
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset

        runner = ResponseClassificationLangfuseRunner("test_dataset")

        # Create 2 valid item results
        valid_item_result_1 = MagicMock(spec=ExperimentItemResult)
        valid_item_result_1.output = MagicMock(spec=CopilotRunResult)
        valid_item_result_1.output.response_category = ResponseCategory.COPILOT
        valid_item_result_1.item = MagicMock(spec=ExperimentItem)
        valid_item_result_1.item.id = "valid_item_1"
        valid_item_result_1.item.expected_output = {"response_category": "copilot"}

        valid_item_result_2 = MagicMock(spec=ExperimentItemResult)
        valid_item_result_2.output = MagicMock(spec=CopilotRunResult)
        valid_item_result_2.output.response_category = (
            ResponseCategory.OUT_OF_SCOPE_DETECTION
        )
        valid_item_result_2.item = MagicMock(spec=ExperimentItem)
        valid_item_result_2.item.id = "valid_item_2"
        valid_item_result_2.item.expected_output = {
            "response_category": "out_of_scope_detection"
        }

        # Create 2 invalid item results
        # Item result 1: Invalid output
        invalid_item_result_1 = MagicMock(spec=ExperimentItemResult)
        invalid_item_result_1.output = None
        invalid_item_result_1.item = MagicMock(spec=ExperimentItem)
        invalid_item_result_1.item.id = "invalid_item_1"
        invalid_item_result_1.item.expected_output = {"response_category": "copilot"}

        # Item result 2: Invalid expected output
        invalid_item_result_2 = MagicMock(spec=ExperimentItemResult)
        invalid_item_result_2.output = MagicMock(spec=CopilotRunResult)
        invalid_item_result_2.output.response_category = ResponseCategory.COPILOT
        invalid_item_result_2.item = MagicMock(spec=ExperimentItem)
        invalid_item_result_2.item.id = "invalid_item_2"
        invalid_item_result_2.item.expected_output = None

        item_results: List[ExperimentItemResult] = [
            valid_item_result_1,
            valid_item_result_2,
            invalid_item_result_1,
            invalid_item_result_2,
        ]

        # When
        classification_results, skip_count = (
            runner._create_classification_results_from_dataset_items(item_results)
        )

        # Then
        # Verify that we have 2 classification results (one for each valid item)
        assert len(classification_results) == 2
        assert skip_count == 2
        # Verify that the classification results have the expected structure
        assert classification_results[0].prediction == ResponseCategory.COPILOT
        assert classification_results[0].expected == ResponseCategory.COPILOT
        assert (
            classification_results[1].prediction
            == ResponseCategory.OUT_OF_SCOPE_DETECTION
        )
        assert (
            classification_results[1].expected
            == ResponseCategory.OUT_OF_SCOPE_DETECTION
        )

    @patch(
        "rasa.builder.evaluator.response_classification.langfuse_runner.langfuse.get_client"
    )
    def test_report_run_results(
        self,
        mock_get_client: MagicMock,
        tmp_path,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
        mock_experiment_result: MagicMock,
    ) -> None:
        """Test _report_run_results method."""
        # Given
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset

        runner = ResponseClassificationLangfuseRunner("test_dataset", str(tmp_path))

        # When
        runner._report_run_results_to_txt_file(mock_experiment_result)

        # Then
        mock_experiment_result.format.assert_called_once()

        # Check that a file was created with the expected pattern
        files = list(tmp_path.glob("*_run_results.txt"))
        assert len(files) == 1
        assert files[0].name.endswith("_run_results.txt")

    @patch(
        "rasa.builder.evaluator.response_classification.langfuse_runner.langfuse.get_client"
    )
    def test_report_yaml_structured_results(
        self,
        mock_get_client: MagicMock,
        tmp_path,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
    ) -> None:
        """Test _report_yaml_structured_results method."""
        # Given
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset

        runner = ResponseClassificationLangfuseRunner("test_dataset", str(tmp_path))

        # Create mock evaluations
        evaluations = [
            Evaluation(name="test_metric_1", value=0.85, comment="Test metric 1"),
            Evaluation(name="test_metric_2", value=0.90, comment="Test metric 2"),
        ]

        # When
        runner._report_yaml_structured_results(evaluations)

        # Then
        # Check that a YAML file was created with the expected pattern
        files = list(tmp_path.glob("*_run_results.yaml"))
        assert len(files) == 1
        assert files[0].name.endswith("_run_results.yaml")
