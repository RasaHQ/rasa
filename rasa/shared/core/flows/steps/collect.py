from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Text, Union

import structlog

from rasa.shared.constants import ACTION_ASK_PREFIX, UTTER_ASK_PREFIX
from rasa.shared.core.flows.flow_step import FlowStep
from rasa.shared.core.slots import SlotRejection
from rasa.shared.exceptions import RasaException

DEFAULT_ASK_BEFORE_FILLING = False
DEFAULT_RESET_AFTER_FLOW_ENDS = True
DEFAULT_FORCE_SLOT_FILLING = False

logger = structlog.get_logger(__name__)

SilenceTimeoutInstructionType = Union[int, float, Dict[str, Any]]


@dataclass
class CollectInformationFlowStep(FlowStep):
    """A flow step for asking the user for information to fill a specific slot."""

    collect: str
    """The collect information of the flow step."""
    utter: str
    """The utterance that the assistant uses to ask for the slot."""
    collect_action: str
    """The action that the assistant uses to ask for the slot."""
    rejections: List[SlotRejection]
    """how the slot value is validated using predicate evaluation."""
    ask_before_filling: bool = DEFAULT_ASK_BEFORE_FILLING
    """Whether to always ask the question even if the slot is already filled."""
    reset_after_flow_ends: bool = DEFAULT_RESET_AFTER_FLOW_ENDS
    """Whether to reset the slot value at the end of the flow."""
    force_slot_filling: bool = False
    """Whether to keep only the SetSlot command for the collected slot."""
    silence_timeout: Optional[float] = None
    """The silence timeout for the collect information step."""

    @classmethod
    def from_json(
        cls, flow_id: Text, data: Dict[str, Any]
    ) -> CollectInformationFlowStep:
        """Create a CollectInformationFlowStep object from serialized data.

        Args:
            flow_id: The id of the flow that contains the step.
            data: data for a CollectInformationFlowStep object in a serialized format

        Returns:
            A CollectInformationFlowStep object
        """

        silence_timeout = cls._deserialise_silence_timeout(
            data.get("silence_timeout", None)
        )

        base = super().from_json(flow_id, data)
        return CollectInformationFlowStep(
            collect=data["collect"],
            utter=data.get("utter", cls._default_utter(data["collect"])),
            # as of now it is not possible to define a different name for the
            # action, always use the default name 'action_ask_<slot_name>'
            collect_action=f"{ACTION_ASK_PREFIX}{data['collect']}",
            ask_before_filling=data.get("ask_before_filling", False),
            reset_after_flow_ends=data.get("reset_after_flow_ends", True),
            rejections=[
                SlotRejection.from_dict(rejection)
                for rejection in data.get("rejections", [])
            ],
            force_slot_filling=data.get("force_slot_filling", False),
            silence_timeout=silence_timeout,
            **base.__dict__,
        )

    @staticmethod
    def _deserialise_silence_timeout(
        silence_timeout_json: Optional[SilenceTimeoutInstructionType],
    ) -> Optional[float]:
        """Deserialize silence timeout from JSON."""
        if not silence_timeout_json:
            return None

        if not isinstance(silence_timeout_json, (int, float)):
            raise RasaException(
                f"Invalid silence timeout value: {silence_timeout_json}. "
                "If defined at collect step, silence timeout must be a number."
            )

        silence_timeout = silence_timeout_json

        if silence_timeout and silence_timeout < 0:
            raise RasaException(
                f"Invalid silence timeout value: {silence_timeout}. "
                "Silence timeout must be a non-negative number."
            )
        return silence_timeout

    @staticmethod
    def _default_utter(collect: str) -> str:
        return f"{UTTER_ASK_PREFIX}{collect}"

    def as_json(
        self, step_properties: Optional[Dict[Text, Any]] = None
    ) -> Dict[str, Any]:
        """Serialize the CollectInformationFlowStep object.

        Returns:
            the CollectInformationFlowStep object as serialized data
        """
        data = super().as_json()
        data["collect"] = self.collect
        data["utter"] = self.utter
        data["ask_before_filling"] = self.ask_before_filling
        data["reset_after_flow_ends"] = self.reset_after_flow_ends
        data["rejections"] = [rejection.as_dict() for rejection in self.rejections]
        data["force_slot_filling"] = self.force_slot_filling
        if self.silence_timeout:
            data["silence_timeout"] = self.silence_timeout

        return super().as_json(step_properties=data)

    @property
    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return f"collect_{self.collect}"

    @property
    def utterances(self) -> Set[str]:
        """Return all the utterances used in this step."""
        return {self.utter} | {r.utter for r in self.rejections}

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return (
                self.collect == other.collect
                and self.utter == other.utter
                and self.collect_action == other.collect_action
                and self.rejections == other.rejections
                and self.ask_before_filling == other.ask_before_filling
                and self.reset_after_flow_ends == other.reset_after_flow_ends
                and self.force_slot_filling == other.force_slot_filling
                and self.silence_timeout == other.silence_timeout
                and super().__eq__(other)
            )
        return False
