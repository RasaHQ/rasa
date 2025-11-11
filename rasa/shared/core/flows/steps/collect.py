from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Text, Union

import structlog

from rasa.shared.constants import ACTION_ASK_PREFIX, UTTER_ASK_PREFIX
from rasa.shared.core.constants import GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE
from rasa.shared.core.flows.flow_step import FlowStep
from rasa.shared.core.slots import SlotRejection
from rasa.shared.exceptions import RasaException

DEFAULT_ASK_BEFORE_FILLING = False
DEFAULT_RESET_AFTER_FLOW_ENDS = True
DEFAULT_FORCE_SLOT_FILLING = False

logger = structlog.get_logger(__name__)

SilenceTimeoutInstructionType = Union[int, float, Dict[str, float]]


class SilenceTimeout(ABC):
    @abstractmethod
    def get_silence_for_channel(self, channel_name: str) -> float:
        pass

    @abstractmethod
    def to_json(self) -> Dict[str, Any]:
        pass


class SingleSilenceTimeout(SilenceTimeout):
    def __init__(
        self,
        silence_timeout: float,
    ):
        SingleSilenceTimeout._validate(silence_timeout)
        self.silence_timeout = silence_timeout

    def get_silence_for_channel(self, channel_name: str) -> float:
        return self.silence_timeout

    def to_json(self) -> Dict[str, Any]:
        return {
            "silence_timeout": self.silence_timeout,
        }

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SingleSilenceTimeout):
            return False
        return self.silence_timeout == other.silence_timeout

    @staticmethod
    def _validate(silence_timeout: float) -> None:
        if silence_timeout and silence_timeout < 0:
            raise RasaException(
                f"Invalid silence timeout value: {silence_timeout}. "
                "Silence timeout must be a non-negative number."
            )

    @classmethod
    def from_json(cls, silence_timeout: float) -> SingleSilenceTimeout:
        SingleSilenceTimeout._validate(silence_timeout)
        return cls(silence_timeout)


class PerChannelSilenceTimeout(SilenceTimeout):
    def __init__(
        self,
        channel_silence_timeouts: Dict[str, float],
    ):
        self.silence_timeouts = channel_silence_timeouts

    def get_silence_for_channel(self, channel_name: str) -> float:
        return self.silence_timeouts.get(
            channel_name, GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "silence_timeout": self.silence_timeouts,
        }

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PerChannelSilenceTimeout):
            return False
        return self.silence_timeouts == other.silence_timeouts

    @staticmethod
    def _validate(silence_timeout_json: Dict[str, Any]) -> None:
        for channel, timeout in silence_timeout_json.items():
            if not isinstance(timeout, (int, float)) or timeout < 0:
                raise RasaException(
                    f"Invalid silence timeout value: {timeout} for "
                    f"channel '{channel}'. "
                    "If defined at collect step, silence timeout "
                    "must be a non-negative number."
                )

    @classmethod
    def from_json(
        cls, channel_silence_timeouts: Dict[str, Any]
    ) -> PerChannelSilenceTimeout:
        PerChannelSilenceTimeout._validate(channel_silence_timeouts)
        return cls(channel_silence_timeouts)


@dataclass
class DTMFConfig:
    """DTMF configuration for collect steps."""

    length: Optional[int] = None
    """Maximum number of DTMF digits to accept."""
    finish_on_key: Optional[Text] = None
    """Key that indicates the end of DTMF input."""
    allow_audio_input: bool = True
    """Whether to allow audio input alongside DTMF."""

    @classmethod
    def from_json(cls, data: Dict[Text, Any]) -> DTMFConfig:
        """Cannot have both length and finish_on_key."""
        if "length" in data and "finish_on_key" in data:
            raise RasaException(
                "DTMF configuration cannot have both 'length' and 'finish_on_key'."
            )
        return cls(
            length=data.get("length"),
            finish_on_key=data.get("finish_on_key"),
            allow_audio_input=data.get("allow_audio_input", True),
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DTMFConfig):
            return (
                self.length == other.length
                and self.finish_on_key == other.finish_on_key
                and self.allow_audio_input == other.allow_audio_input
            )
        return False


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
    silence_timeout: Optional[SilenceTimeout] = None
    """The silence timeout for the collect information step."""
    dtmf: Optional[DTMFConfig] = None
    """DTMF configuration for the collect information step."""

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

        dtmf = None
        if "dtmf" in data:
            dtmf = DTMFConfig.from_json(data["dtmf"])

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
            dtmf=dtmf,
            **base.__dict__,
        )

    @staticmethod
    def _deserialise_silence_timeout(
        silence_timeout_json: Optional[SilenceTimeoutInstructionType],
    ) -> Optional[SilenceTimeout]:
        """Deserialize silence timeout from JSON."""
        if not silence_timeout_json:
            return None

        if not isinstance(silence_timeout_json, (int, float, dict)):
            raise RasaException(
                f"Invalid silence timeout value: {silence_timeout_json}. "
                "If defined at collect step, silence timeout must be a number "
                "or a map between channel names and timeout values."
            )

        if isinstance(silence_timeout_json, dict):
            return PerChannelSilenceTimeout.from_json(silence_timeout_json)

        return SingleSilenceTimeout.from_json(silence_timeout_json)

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
            data.update(self.silence_timeout.to_json())

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
                and self.dtmf == other.dtmf
                and super().__eq__(other)
            )
        return False
