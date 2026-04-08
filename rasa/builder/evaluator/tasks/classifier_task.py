"""Task that runs the MessageClassifier in isolation.

This is the lightweight alternative to copilot_task — it calls only the
classifier, skipping response generation entirely. Use this when you want
fast iteration on the classifier prompt or model without paying for
full copilot execution.
"""

from typing import Any, Optional

import structlog

from rasa.builder.evaluator.dataset.models import DatasetEntry
from rasa.builder.evaluator.tasks.base import BaseTask, TaskResult

structlogger = structlog.get_logger()


class ClassifierTask(BaseTask):
    """Callable task that runs the MessageClassifier on a single dataset item.

    The classifier is instantiated once in ``__init__`` and reused across all
    items, avoiding redundant initialisation overhead per item.
    """

    def __init__(self) -> None:
        from rasa.builder.copilot.message_classifier.message_classifier import (
            MessageClassifier,
        )

        self._classifier = MessageClassifier()

    async def run_task(
        self,
        *,
        item: Any,
        **_: Any,
    ) -> Optional[TaskResult]:
        """Run the MessageClassifier on a dataset item.

        Args:
            item: Langfuse ExperimentItem.
            kwargs: Additional keyword arguments passed by Langfuse.

        Returns:
            TaskResult with predicted category, or None on failure.
        """
        try:
            dataset_entry = DatasetEntry.from_raw_data(
                id=item.id,
                input_data=item.input,
                expected_output_data=item.expected_output,
                metadata_data=item.metadata,
            )
            context = dataset_entry.to_copilot_context()
            result = await self._classifier.classify(context)

            return TaskResult(
                predicted_category=result.category,
                complete_response=result.raw_response,
            )
        except Exception as e:
            structlogger.error(
                "tasks.classifier_task.failed",
                event_info=f"Classification failed for item {item.id}.",
                error=str(e),
            )
            return None
