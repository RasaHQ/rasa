"""Base models and abstract class for experiment tasks."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel

from rasa.builder.copilot.models import ResponseCategory


class TaskResult(BaseModel):
    """Standardized result from any experiment task.

    Attributes:
        predicted_category: The category predicted by the task.
        complete_response: The full text response (if applicable).
    """

    predicted_category: Optional[ResponseCategory]
    complete_response: Optional[str] = None


class BaseTask(ABC):
    """Contract every task must follow.

    Subclasses implement ``run_task`` which is called once per dataset item
    by Langfuse's ``dataset.run_experiment()``.
    """

    @abstractmethod
    async def run_task(self, *, item: Any, **kwargs: Any) -> Optional[TaskResult]:
        """Execute the task on a single dataset item.

        Args:
            item: Langfuse ExperimentItem.
            kwargs: Additional keyword arguments passed by Langfuse.

        Returns:
            TaskResult on success, or None on failure.
        """
        ...
