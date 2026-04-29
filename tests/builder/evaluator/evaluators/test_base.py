"""Tests for BaseEvaluator template method."""

from typing import Any, List, Tuple

# Import after require_langfuse is handled by builder conftest autouse fixtures
from langfuse import Evaluation  # noqa: TID251
from langfuse.experiment import ExperimentItemResult  # noqa: TID251

from rasa.builder.evaluator.evaluators.base import BaseEvaluator


class _StubEvaluator(BaseEvaluator):
    """Concrete evaluator for testing the base template method."""

    def __init__(
        self,
        extract_return: Any = (["result"], 0),
        evaluate_return: Any = "summary",
        evaluations_return: Any = None,
        raise_in: str | None = None,
    ):
        super().__init__()
        self._extract_return = extract_return
        self._evaluate_return = evaluate_return
        self._evaluations_return = evaluations_return or [
            Evaluation(name="test", value=1.0)
        ]
        self._raise_in = raise_in

    def extract_results(
        self, item_results: List[ExperimentItemResult]
    ) -> Tuple[Any, int]:
        if self._raise_in == "extract_results":
            raise RuntimeError("extract boom")
        return self._extract_return

    def evaluate(self, results: Any) -> Any:
        if self._raise_in == "evaluate":
            raise RuntimeError("evaluate boom")
        return self._evaluate_return

    def to_evaluations(self, summary: Any, skip_count: int) -> List[Evaluation]:
        if self._raise_in == "to_evaluations":
            raise RuntimeError("to_evaluations boom")
        return self._evaluations_return


class TestBaseEvaluatorRun:
    def test_success_pipeline(self):
        evaluator = _StubEvaluator()
        result = evaluator.run(item_results=[])

        assert len(result) == 1
        assert result[0].name == "test"
        assert evaluator.results == ["result"]
        assert evaluator.summary == "summary"

    def test_exception_returns_empty_list(self):
        evaluator = _StubEvaluator(raise_in="extract_results")
        result = evaluator.run(item_results=[])

        assert result == []

    def test_exception_in_evaluate_returns_empty(self):
        evaluator = _StubEvaluator(raise_in="evaluate")
        result = evaluator.run(item_results=[])

        assert result == []
        assert evaluator.results is None
        assert evaluator.summary is None

    def test_init_defaults(self):
        evaluator = _StubEvaluator()

        assert evaluator.results is None
        assert evaluator.summary is None
