"""Base models and abstract class for experiment tasks."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from pydantic import BaseModel

from rasa.builder.copilot.models import ResponseCategory
from rasa.builder.evaluator.configs.models import ExperimentConfig


class ClassifierTaskResult(BaseModel):
    """Standardized result from a classification task.

    Attributes:
        predicted_category: The category predicted by the task.
        complete_response: The full text response (if applicable).
    """

    predicted_category: Optional[ResponseCategory]
    complete_response: Optional[str] = None


class RetrievalTaskResult(BaseModel):
    """Result from a retrieval task.

    Attributes:
        query: The query that was executed.
        retrieved_urls: Ordered list of URLs returned by the retriever.
        retrieved_titles: Titles parallel to retrieved_urls.
        latency_ms: Wall-clock time of the retrieval call.
        error: Populated if the retrieval call failed.
    """

    query: str = ""
    retrieved_urls: List[str] = []
    retrieved_titles: List[str] = []
    latency_ms: float = 0.0
    error: Optional[str] = None


class BaseTask(ABC):
    """Contract every task must follow.

    Subclasses implement ``run_task`` which is called once per dataset item
    by Langfuse's ``dataset.run_experiment()``.

    The ``__init__`` receives the full ``ExperimentConfig`` so each task can
    read task-specific fields (e.g. ``config.retrieval``) if present.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    @abstractmethod
    async def run_task(self, *, item: Any, **kwargs: Any) -> Optional[Any]:
        """Execute the task on a single dataset item.

        Args:
            item: Langfuse ExperimentItem.
            kwargs: Additional keyword arguments passed by Langfuse.

        Returns:
            A task-specific result model, or None on failure.
        """
        ...
