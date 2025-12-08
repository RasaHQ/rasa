"""Tests for CopilotResponseEvaluatorLangfuseRunner."""

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from langfuse import Evaluation
from langfuse._client.datasets import DatasetClient
from langfuse.experiment import ExperimentResult

from rasa.builder.copilot.models import ReferenceEntry, ResponseCategory
from rasa.builder.evaluator.completeness_judge.models import (
    CompletenessJudgeResult,
    UserRequestCompletenessVerdict,
)
from rasa.builder.evaluator.content_processors.models import (
    Claim,
    ClaimImportance,
    Claims,
)
from rasa.builder.evaluator.copilot_response_evaluator.langfuse_runner import (
    CopilotResponseEvaluatorLangfuseRunner,
)
from rasa.builder.evaluator.copilot_response_evaluator.models import (
    CopilotResponseCompletenessEvaluationResult,
    CopilotResponseFaithfulnessEvaluationResult,
)
from rasa.builder.evaluator.faithfulness_judge.models import (
    ClaimVerdict,
    FaithfulnessJudgeResult,
    FaithfulnessVerdictLabel,
)
from rasa.builder.evaluator.shared.copilot_executor import CopilotRunResult
from rasa.builder.evaluator.shared.models import EvaluationFailure


class TestCopilotResponseEvaluatorLangfuseRunner:
    """Test suite for CopilotResponseEvaluatorLangfuseRunner."""

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
        result.dataset_run_url = "https://langfuse.test/run"
        result.dataset_run_id = "run-123"
        result.item_results = []
        return result

    @pytest.fixture
    def mock_copilot_run_result(self) -> MagicMock:
        """Mock CopilotRunResult."""
        mock_result = MagicMock(spec=CopilotRunResult)
        mock_result.complete_response = "Test response"
        mock_result.response_category = ResponseCategory.COPILOT
        mock_result.reference_section = MagicMock()
        mock_result.reference_section.references = [
            ReferenceEntry(index=0, title="Test Doc", url="https://example.com/doc")
        ]
        return mock_result

    @pytest.fixture
    def runner_with_mock_langfuse(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
        tmp_path: Path,
    ) -> CopilotResponseEvaluatorLangfuseRunner:
        """Provide a runner instance with Langfuse client mocked via monkeypatch."""
        mock_langfuse_client.get_dataset.return_value = mock_dataset
        monkeypatch.setattr(
            "rasa.builder.evaluator.copilot_response_evaluator.langfuse_runner.langfuse.get_client",
            MagicMock(return_value=mock_langfuse_client),
        )
        return CopilotResponseEvaluatorLangfuseRunner("test_dataset", str(tmp_path))

    @pytest.fixture
    def sample_faithfulness_result(self) -> CopilotResponseFaithfulnessEvaluationResult:
        """Create a sample faithfulness evaluation result."""
        claims = Claims(
            claims=[
                Claim(
                    id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                    importance=ClaimImportance.HIGH,
                    text="Test claim 1",
                    metadata={},
                ),
                Claim(
                    id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                    importance=ClaimImportance.MEDIUM,
                    text="Test claim 2",
                    metadata={},
                ),
            ]
        )
        faithfulness_judge_result = FaithfulnessJudgeResult(
            verdicts=[
                ClaimVerdict(
                    claim_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                    verdict=FaithfulnessVerdictLabel.SUPPORTED,
                    rationale="Supported by evidence",
                    confidence=0.9,
                ),
                ClaimVerdict(
                    claim_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                    verdict=FaithfulnessVerdictLabel.SUPPORTED,
                    rationale="Supported by evidence",
                    confidence=0.8,
                ),
            ]
        )
        return CopilotResponseFaithfulnessEvaluationResult(
            entry_id="test-entry-1",
            extracted_claims=claims,
            faithfulness_result=faithfulness_judge_result,
        )

    @pytest.fixture
    def sample_completeness_result(
        self,
    ) -> CopilotResponseCompletenessEvaluationResult:
        """Create a sample completeness evaluation result."""
        claims = Claims(
            claims=[
                Claim(
                    id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                    importance=ClaimImportance.HIGH,
                    text="Test claim 1",
                    metadata={},
                ),
            ]
        )
        completeness_judge_result = CompletenessJudgeResult(
            verdicts=[
                UserRequestCompletenessVerdict(
                    part_text="How to configure X",
                    addressing_claims_ids=[
                        uuid.UUID("00000000-0000-0000-0000-000000000001")
                    ],
                    rationale="Question fully addressed",
                ),
            ],
            overall_rationale="Complete response",
            confidence=0.9,
        )
        return CopilotResponseCompletenessEvaluationResult(
            entry_id="test-entry-1",
            extracted_claims=claims,
            completeness_result=completeness_judge_result,
        )

    @patch(
        "rasa.builder.evaluator.copilot_response_evaluator.langfuse_runner.langfuse.get_client"
    )
    def test_run_experiment(
        self,
        mock_get_client: MagicMock,
        tmp_path: Path,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
        mock_experiment_result: MagicMock,
    ) -> None:
        """Test run_experiment method."""
        # Given
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset
        mock_dataset.run_experiment.return_value = mock_experiment_result

        runner = CopilotResponseEvaluatorLangfuseRunner("test_dataset", str(tmp_path))

        # When
        result = runner.run_experiment()

        # Then
        assert result == mock_experiment_result
        mock_dataset.run_experiment.assert_called_once()
        call_args = mock_dataset.run_experiment.call_args
        assert call_args[1]["name"] == "Copilot Response Quality Evaluation"
        assert "description" in call_args[1]
        assert "task" in call_args[1]
        assert "evaluators" in call_args[1]
        mock_langfuse_client.flush.assert_called_once()

    @patch(
        "rasa.builder.evaluator.copilot_response_evaluator.langfuse_runner.langfuse.get_client"
    )
    def test_retrieve_dataset_success(
        self,
        mock_get_client: MagicMock,
        tmp_path: Path,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
    ) -> None:
        """Test _retrieve_dataset with successful dataset retrieval."""
        # Given
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset

        runner = CopilotResponseEvaluatorLangfuseRunner("test_dataset", str(tmp_path))

        # Then
        assert runner._dataset == mock_dataset
        mock_langfuse_client.get_dataset.assert_called_once_with("test_dataset")

    @patch(
        "rasa.builder.evaluator.copilot_response_evaluator.langfuse_runner.langfuse.get_client"
    )
    def test_retrieve_dataset_failure(
        self,
        mock_get_client: MagicMock,
        tmp_path: Path,
        mock_langfuse_client: MagicMock,
    ) -> None:
        """Test _retrieve_dataset when dataset retrieval fails."""
        # Given
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.side_effect = Exception("Dataset not found")

        # When/Then
        with pytest.raises(Exception, match="Dataset not found"):
            CopilotResponseEvaluatorLangfuseRunner("test_dataset", str(tmp_path))

    @pytest.mark.asyncio
    @patch(
        "rasa.builder.evaluator.copilot_response_evaluator.langfuse_runner.CopilotResponseEvaluator.evaluate"
    )
    @patch(
        "rasa.builder.evaluator.copilot_response_evaluator.langfuse_runner.langfuse.get_client"
    )
    async def test_response_quality_evaluator_success(
        self,
        mock_get_client: MagicMock,
        mock_evaluator_evaluate: AsyncMock,
        tmp_path: Path,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
        mock_copilot_run_result: MagicMock,
        sample_faithfulness_result: CopilotResponseFaithfulnessEvaluationResult,
        sample_completeness_result: CopilotResponseCompletenessEvaluationResult,
    ) -> None:
        """Test _response_quality_evaluator with successful evaluation."""
        # Given
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset

        mock_evaluator_evaluate.return_value = (
            [sample_faithfulness_result],
            [sample_completeness_result],
        )

        runner = CopilotResponseEvaluatorLangfuseRunner("test_dataset", str(tmp_path))

        input_data = {"message": "How do I configure X?"}
        metadata = {
            "copilot_additional_context": {
                "assistant_logs": "test logs",
                "relevant_assistant_files": {},
                "assistant_tracker_context": None,
                "copilot_chat_history": [],
            }
        }

        # When
        evaluations = await runner._response_quality_evaluator(
            input=input_data,
            output=mock_copilot_run_result,
            expected_output=None,
            metadata=metadata,
        )

        # Then
        assert isinstance(evaluations, list)
        assert all(isinstance(eval_obj, Evaluation) for eval_obj in evaluations)
        assert len(evaluations) > 0

        # Verify faithfulness metrics are present
        faithfulness_metric_names = {eval_obj.name for eval_obj in evaluations}
        assert "faithfulness_support_rate" in faithfulness_metric_names
        assert "faithfulness_avg_verdict_confidence" in faithfulness_metric_names
        assert "faithfulness_supported_claims_count" in faithfulness_metric_names
        assert "faithfulness_extracted_claims_count" in faithfulness_metric_names

        # Verify completeness metrics are present
        completeness_metric_names = {eval_obj.name for eval_obj in evaluations}
        assert "completeness_coverage_rate" in completeness_metric_names
        assert "completeness_confidence" in completeness_metric_names
        assert "completeness_covered_parts" in completeness_metric_names

    @pytest.mark.asyncio
    @patch(
        "rasa.builder.evaluator.copilot_response_evaluator.langfuse_runner.langfuse.get_client"
    )
    async def test_response_quality_evaluator_dataset_entry_creation_failure(
        self,
        mock_get_client: MagicMock,
        tmp_path: Path,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
        mock_copilot_run_result: MagicMock,
    ) -> None:
        """Test _response_quality_evaluator when dataset entry creation fails."""
        # Given
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset

        runner = CopilotResponseEvaluatorLangfuseRunner("test_dataset", str(tmp_path))

        # Invalid input that will cause validation to fail
        invalid_input = None
        metadata = None

        # When
        evaluations = await runner._response_quality_evaluator(
            input=invalid_input,
            output=mock_copilot_run_result,
            expected_output=None,
            metadata=metadata,
        )

        # Then
        assert evaluations == []

    @pytest.mark.asyncio
    @patch(
        "rasa.builder.evaluator.copilot_response_evaluator.langfuse_runner.CopilotResponseEvaluator.evaluate"
    )
    @patch(
        "rasa.builder.evaluator.copilot_response_evaluator.langfuse_runner.langfuse.get_client"
    )
    async def test_response_quality_evaluator_evaluation_failure(
        self,
        mock_get_client: MagicMock,
        mock_evaluator_evaluate: AsyncMock,
        tmp_path: Path,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
        mock_copilot_run_result: MagicMock,
    ) -> None:
        """Test _response_quality_evaluator when evaluation fails."""
        # Given
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset

        mock_evaluator_evaluate.side_effect = Exception("Evaluation failed")

        runner = CopilotResponseEvaluatorLangfuseRunner("test_dataset", str(tmp_path))

        input_data = {"message": "How do I configure X?"}
        metadata = {
            "copilot_additional_context": {
                "assistant_logs": "test logs",
                "relevant_assistant_files": {},
                "assistant_tracker_context": None,
                "copilot_chat_history": [],
            }
        }

        # When
        evaluations = await runner._response_quality_evaluator(
            input=input_data,
            output=mock_copilot_run_result,
            expected_output=None,
            metadata=metadata,
        )

        # Then
        assert evaluations == []

    def test_parse_faithfulness_evaluation_results_success(
        self,
        sample_faithfulness_result: CopilotResponseFaithfulnessEvaluationResult,
        runner_with_mock_langfuse: CopilotResponseEvaluatorLangfuseRunner,
    ) -> None:
        """Test _parse_faithfulness_evaluation_results with successful result."""
        # Given
        runner = runner_with_mock_langfuse

        # When
        evaluations = runner._parse_faithfulness_evaluation_results(
            sample_faithfulness_result
        )

        # Then
        assert isinstance(evaluations, list)
        assert len(evaluations) == 7  # 2 agg + reasoning + 4 info metrics

        # Verify all expected metrics are present
        metric_names = {eval_obj.name for eval_obj in evaluations}
        assert "faithfulness_support_rate" in metric_names
        assert "faithfulness_avg_verdict_confidence" in metric_names
        assert "faithfulness_supported_claims_count" in metric_names
        assert "faithfulness_contradicted_claims_count" in metric_names
        assert "faithfulness_not_enough_info_claims_count" in metric_names
        assert "faithfulness_extracted_claims_count" in metric_names

        # Verify metric values
        support_rate_eval = next(
            e for e in evaluations if e.name == "faithfulness_support_rate"
        )
        assert support_rate_eval.value == 1.0  # 2 supported / 2 total

        supported_count_eval = next(
            e for e in evaluations if e.name == "faithfulness_supported_claims_count"
        )
        assert supported_count_eval.value == 2.0

        extracted_count_eval = next(
            e for e in evaluations if e.name == "faithfulness_extracted_claims_count"
        )
        assert extracted_count_eval.value == 2.0

    def test_parse_faithfulness_evaluation_results_with_failure(
        self,
        runner_with_mock_langfuse: CopilotResponseEvaluatorLangfuseRunner,
    ) -> None:
        """Test _parse_faithfulness_evaluation_results with evaluation failure."""
        # Given
        failure = EvaluationFailure(
            error_type="OpenAIError", error_message="API call failed"
        )
        claims = Claims(claims=[])
        result = CopilotResponseFaithfulnessEvaluationResult(
            entry_id="test-entry-1",
            extracted_claims=claims,
            faithfulness_result=failure,
        )

        runner = runner_with_mock_langfuse

        # When
        evaluations = runner._parse_faithfulness_evaluation_results(result)

        # Then
        assert isinstance(evaluations, list)
        assert len(evaluations) == 7  # Still returns all metrics with zero values

        # Verify metrics are zero for failure case
        support_rate_eval = next(
            e for e in evaluations if e.name == "faithfulness_support_rate"
        )
        assert support_rate_eval.value == 0.0

        extracted_count_eval = next(
            e for e in evaluations if e.name == "faithfulness_extracted_claims_count"
        )
        assert extracted_count_eval.value == 0.0

    def test_parse_completeness_evaluation_results_success(
        self,
        sample_completeness_result: CopilotResponseCompletenessEvaluationResult,
        runner_with_mock_langfuse: CopilotResponseEvaluatorLangfuseRunner,
    ) -> None:
        """Test _parse_completeness_evaluation_results with successful result."""
        # Given
        runner = runner_with_mock_langfuse

        # When
        evaluations = runner._parse_completeness_evaluation_results(
            sample_completeness_result
        )

        # Then
        assert isinstance(evaluations, list)
        assert len(evaluations) == 6  # 2 agg + reasoning + 3 info metrics

        # Verify all expected metrics are present
        metric_names = {eval_obj.name for eval_obj in evaluations}
        assert "completeness_covered_parts" in metric_names
        assert "completeness_uncovered_parts" in metric_names
        assert "completeness_total_parts" in metric_names
        assert "completeness_coverage_rate" in metric_names
        assert "completeness_confidence" in metric_names

        # Verify metric values
        coverage_rate_eval = next(
            e for e in evaluations if e.name == "completeness_coverage_rate"
        )
        assert coverage_rate_eval.value == 1.0  # 1 covered / 1 total

        covered_parts_eval = next(
            e for e in evaluations if e.name == "completeness_covered_parts"
        )
        assert covered_parts_eval.value == 1.0

        total_parts_eval = next(
            e for e in evaluations if e.name == "completeness_total_parts"
        )
        assert total_parts_eval.value == 1.0

        confidence_eval = next(
            e for e in evaluations if e.name == "completeness_confidence"
        )
        assert confidence_eval.value == 0.9

    def test_parse_completeness_evaluation_results_with_failure(
        self,
        runner_with_mock_langfuse: CopilotResponseEvaluatorLangfuseRunner,
    ) -> None:
        """Test _parse_completeness_evaluation_results with evaluation failure."""
        # Given
        failure = EvaluationFailure(
            error_type="OpenAIError", error_message="API call failed"
        )
        claims = Claims(claims=[])
        result = CopilotResponseCompletenessEvaluationResult(
            entry_id="test-entry-1",
            extracted_claims=claims,
            completeness_result=failure,
        )

        runner = runner_with_mock_langfuse

        # When
        evaluations = runner._parse_completeness_evaluation_results(result)

        # Then
        assert isinstance(evaluations, list)
        assert len(evaluations) == 6  # Still returns all metrics with zero values

        # Verify metrics are zero for failure case
        coverage_rate_eval = next(
            e for e in evaluations if e.name == "completeness_coverage_rate"
        )
        assert coverage_rate_eval.value == 0.0

        covered_parts_eval = next(
            e for e in evaluations if e.name == "completeness_covered_parts"
        )
        assert covered_parts_eval.value == 0.0

        confidence_eval = next(
            e for e in evaluations if e.name == "completeness_confidence"
        )
        assert confidence_eval.value == 0.0

    def test_parse_faithfulness_evaluation_results_with_mixed_verdicts(
        self,
        runner_with_mock_langfuse: CopilotResponseEvaluatorLangfuseRunner,
    ) -> None:
        """Test _parse_faithfulness_evaluation_results with mixed verdict types."""
        # Given
        claims = Claims(
            claims=[
                Claim(
                    id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                    importance=ClaimImportance.HIGH,
                    text="Supported claim",
                    metadata={},
                ),
                Claim(
                    id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                    importance=ClaimImportance.MEDIUM,
                    text="Contradicted claim",
                    metadata={},
                ),
                Claim(
                    id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                    importance=ClaimImportance.LOW,
                    text="Not enough info claim",
                    metadata={},
                ),
            ]
        )
        faithfulness_judge_result = FaithfulnessJudgeResult(
            verdicts=[
                ClaimVerdict(
                    claim_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                    verdict=FaithfulnessVerdictLabel.SUPPORTED,
                    rationale="Supported",
                    confidence=0.9,
                ),
                ClaimVerdict(
                    claim_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                    verdict=FaithfulnessVerdictLabel.CONTRADICTED,
                    rationale="Contradicted",
                    confidence=0.8,
                ),
                ClaimVerdict(
                    claim_id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                    verdict=FaithfulnessVerdictLabel.NOT_ENOUGH_INFO,
                    rationale="Not enough info",
                    confidence=0.7,
                ),
            ]
        )
        result = CopilotResponseFaithfulnessEvaluationResult(
            entry_id="test-entry-1",
            extracted_claims=claims,
            faithfulness_result=faithfulness_judge_result,
        )

        runner = runner_with_mock_langfuse

        # When
        evaluations = runner._parse_faithfulness_evaluation_results(result)

        # Then
        supported_count_eval = next(
            e for e in evaluations if e.name == "faithfulness_supported_claims_count"
        )
        assert supported_count_eval.value == 1.0

        contradicted_count_eval = next(
            e for e in evaluations if e.name == "faithfulness_contradicted_claims_count"
        )
        assert contradicted_count_eval.value == 1.0

        not_enough_info_count_eval = next(
            e
            for e in evaluations
            if e.name == "faithfulness_not_enough_info_claims_count"
        )
        assert not_enough_info_count_eval.value == 1.0

        support_rate_eval = next(
            e for e in evaluations if e.name == "faithfulness_support_rate"
        )
        assert support_rate_eval.value == pytest.approx(
            1.0 / 3.0
        )  # 1 supported / 3 total

    @patch(
        "rasa.builder.evaluator.copilot_response_evaluator.langfuse_runner.langfuse.get_client"
    )
    def test_report_run_results_to_txt_file(
        self,
        mock_get_client: MagicMock,
        tmp_path: Path,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
        mock_experiment_result: MagicMock,
    ) -> None:
        """Test _report_run_results_to_txt_file writes timestamped text results."""
        # Given
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset
        mock_experiment_result.format.return_value = "Mock experiment result"

        runner = CopilotResponseEvaluatorLangfuseRunner("test_dataset", str(tmp_path))

        # When
        runner._report_run_results_to_txt_file(mock_experiment_result)

        # Then
        mock_experiment_result.format.assert_called_once_with()
        files = list(tmp_path.glob("*_run_results.txt"))
        assert len(files) == 1
        assert files[0].name.endswith("run_results.txt")

    @patch(
        "rasa.builder.evaluator.copilot_response_evaluator.langfuse_runner.langfuse.get_client"
    )
    def test_report_yaml_structured_results(
        self,
        mock_get_client: MagicMock,
        tmp_path: Path,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
    ) -> None:
        """Test _report_yaml_structured_results exports structured metrics."""
        # Given
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset

        runner = CopilotResponseEvaluatorLangfuseRunner("test_dataset", str(tmp_path))

        evaluation = Evaluation(
            name="faithfulness_support_rate",
            value=0.75,
            comment="Support rate",
            metadata={"details": {"claim_id": "claim-1"}},
        )
        mock_item = MagicMock()
        mock_item.id = "item-123"
        mock_item.input = {"message": "Hello"}

        mock_output = MagicMock()
        mock_output.complete_response = "Test response"

        mock_item_result = MagicMock()
        mock_item_result.item = mock_item
        mock_item_result.output = mock_output
        mock_item_result.evaluations = [evaluation]

        mock_experiment_result = MagicMock(spec=ExperimentResult)
        mock_experiment_result.dataset_run_url = "https://langfuse.test/run"
        mock_experiment_result.dataset_run_id = "run-123"
        mock_experiment_result.item_results = [mock_item_result]

        # When
        runner._report_yaml_structured_results(mock_experiment_result)

        # Then
        files = list(tmp_path.glob("*_run_results.yaml"))
        assert len(files) == 1
        output_path = files[0]
        data = yaml.safe_load(output_path.read_text())
        assert data["experiment"]["run_id"] == "run-123"
        assert data["experiment"]["run_url"] == "https://langfuse.test/run"
        assert data["per_item_metrics"][0]["id"] == "item-123"
        assert (
            data["per_item_metrics"][0]["evaluations"][0]["name"]
            == "faithfulness_support_rate"
        )
        assert data["per_item_metrics"][0]["evaluations"][0]["metadata"] == {
            "details": {"claim_id": "claim-1"}
        }

    @patch(
        "rasa.builder.evaluator.copilot_response_evaluator.langfuse_runner.langfuse.get_client"
    )
    def test_run_experiment_reporting_failure_does_not_raise(
        self,
        mock_get_client: MagicMock,
        tmp_path: Path,
        mock_langfuse_client: MagicMock,
        mock_dataset: MagicMock,
        mock_experiment_result: MagicMock,
    ) -> None:
        """Ensure run_experiment swallows reporting errors and logs them."""
        mock_get_client.return_value = mock_langfuse_client
        mock_langfuse_client.get_dataset.return_value = mock_dataset
        mock_dataset.run_experiment.return_value = mock_experiment_result

        runner = CopilotResponseEvaluatorLangfuseRunner("test_dataset", str(tmp_path))

        with patch.object(
            CopilotResponseEvaluatorLangfuseRunner,
            "_report_run_results_to_txt_file",
            side_effect=Exception("report failed"),
        ):
            result = runner.run_experiment()

        assert result == mock_experiment_result
        mock_langfuse_client.flush.assert_called_once()
