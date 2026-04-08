"""Abstract base class for experiment evaluators."""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import structlog

from rasa.builder.telemetry.langfuse.langfuse_compat import require_langfuse

require_langfuse()

from langfuse import Evaluation  # noqa: E402, TID251
from langfuse.experiment import ExperimentItemResult  # noqa: E402, TID251

structlogger = structlog.get_logger()


class BaseEvaluator(ABC):
    """Contract every evaluator must follow.

    Subclasses implement three abstract methods for the evaluation pipeline:
        extract_results → evaluate → to_evaluations

    The ``run`` template method orchestrates these steps and conforms to
    Langfuse's RunEvaluatorFunction protocol, so it can be passed directly
    as ``run_evaluators`` or ``evaluators`` to ``dataset.run_experiment()``.

    After ``run`` completes, intermediate data is available via:
        - ``results``: parsed domain-specific results from ``extract_results``
        - ``summary``: computed metrics from ``evaluate``
    """

    def __init__(self) -> None:
        self.results: Any = None
        self.summary: Any = None

    @abstractmethod
    def extract_results(
        self, item_results: List[ExperimentItemResult]
    ) -> Tuple[Any, int]:
        """Parse Langfuse item results into domain-specific results.

        Returns:
            Tuple of (parsed_results, skip_count).
        """
        ...

    @abstractmethod
    def evaluate(self, results: Any) -> Any:
        """Compute metrics from parsed results.

        Returns:
            A evaluator-specific type.
        """
        ...

    @abstractmethod
    def to_evaluations(self, summary: Any, skip_count: int) -> List[Evaluation]:
        """Convert the evaluator results into Langfuse Evaluation objects."""
        ...

    def run(
        self, *, item_results: List[ExperimentItemResult], **kwargs: Any
    ) -> List[Evaluation]:
        """Template method — the Langfuse entry point.

        Subclasses should not override this. They implement the three
        abstract methods; this method orchestrates them.
        """
        try:
            results, skip_count = self.extract_results(item_results)
            summary = self.evaluate(results)
            self.results = results
            self.summary = summary
            return self.to_evaluations(summary, skip_count)
        except Exception as e:
            structlogger.error(
                "evaluators.base.run.failed",
                evaluator=type(self).__name__,
                error=str(e),
            )
            return []
