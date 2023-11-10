from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rasa.shared.core.events import Event


@dataclass
class FlowActionPrediction:
    """Represents an action prediction."""

    action_name: Optional[str]
    """The name of the predicted action."""
    score: float
    """The score of the predicted action."""
    metadata: Optional[Dict[str, Any]] = None
    """The metadata of the predicted action."""
    events: Optional[List[Event]] = None
    """The events attached to the predicted action."""


class FlowStepResult:
    """Represents the result of a flow step."""

    def __init__(self, events: Optional[List[Event]] = None) -> None:
        self.events = events or []


class ContinueFlowWithNextStep(FlowStepResult):
    """Represents the result of a flow step that should continue with the next step."""

    def __init__(
        self, events: Optional[List[Event]] = None, has_flow_ended: bool = False
    ) -> None:
        self.has_flow_ended = has_flow_ended
        super().__init__(events=events)


class PauseFlowReturnPrediction(FlowStepResult):
    """Result where the flow execution should be paused after this step."""

    def __init__(self, action_prediction: FlowActionPrediction) -> None:
        self.action_prediction = action_prediction
        super().__init__(events=action_prediction.events)
