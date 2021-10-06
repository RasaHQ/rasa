from __future__ import annotations
import logging

from rasa.core.policies.policy import PolicyPrediction
from typing import Dict, Optional, Text, Any, List, Tuple

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.core.trackers import DialogueStateTracker


class PredictionOutputProvider(GraphComponent):
    """Provides the a unified output for model predictions."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> PredictionOutputProvider:
        """Creates component (see parent class for full docstring)."""
        return cls()

    def provide(
        self,
        parsed_messages: Optional[List[Message]] = None,
        tracker_with_added_message: Optional[DialogueStateTracker] = None,
        ensemble_output: Optional[Tuple[DialogueStateTracker, PolicyPrediction]] = None,
    ) -> Tuple[
        Optional[Message], Optional[DialogueStateTracker], Optional[PolicyPrediction]
    ]:
        """Provides the parsed message, tracker and policy prediction if available."""
        parsed_message = parsed_messages[0] if parsed_messages else None

        tracker = tracker_with_added_message

        policy_prediction = None
        if ensemble_output:
            tracker, policy_prediction = ensemble_output

        return parsed_message, tracker, policy_prediction
