from __future__ import annotations

import dataclasses
import json
import re
import sys
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Text,
    Tuple,
    Type,
)

import structlog
from jinja2 import Template

import rasa.shared.utils.common
import rasa.shared.utils.io
from rasa.core.constants import DOMAIN_GROUND_TRUTH_METADATA_KEY
from rasa.core.policies.enterprise_search_policy import SEARCH_RESULTS_METADATA_KEY
from rasa.dialogue_understanding.patterns.clarify import FLOW_PATTERN_CLARIFICATION
from rasa.e2e_test.constants import (
    DEFAULT_ANSWER_RELEVANCE_PROMPT_TEMPLATE_FILE_NAME,
    DEFAULT_GROUNDEDNESS_PROMPT_TEMPLATE_FILE_NAME,
    KEY_GROUND_TRUTH,
    KEY_THRESHOLD,
    KEY_UTTER_NAME,
    KEY_UTTER_SOURCE,
    LLM_JUDGE_PROMPTS_MODULE,
)
from rasa.e2e_test.utils.generative_assertions import (
    ScoreInputs,
    _find_matching_generative_events,
    _parse_llm_output,
    _validate_parsed_llm_output,
    calculate_groundedness_score,
    calculate_relevance_score,
)
from rasa.shared.core.constants import DEFAULT_SLOT_NAMES
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    DefinePrevUserUtteredFeaturization,
    DialogueStackUpdated,
    Event,
    FlowCancelled,
    FlowCompleted,
    FlowStarted,
    SlotSet,
)
from rasa.shared.exceptions import ProviderClientAPIException, RasaException
from rasa.shared.utils.llm import (
    llm_factory,
)
from rasa.utils.json_utils import SetEncoder

if TYPE_CHECKING:
    from rasa.e2e_test.e2e_config import LLMJudgeConfig


structlogger = structlog.get_logger()

DEFAULT_THRESHOLD = 0.5
OPERATOR_KEY = "operator"
FLOW_IDS_KEY = "flow_ids"


class AssertionType(Enum):
    FLOW_STARTED = "flow_started"
    FLOW_COMPLETED = "flow_completed"
    FLOW_CANCELLED = "flow_cancelled"
    PATTERN_CLARIFICATION_CONTAINS = "pattern_clarification_contains"
    ACTION_EXECUTED = "action_executed"
    SLOT_WAS_SET = "slot_was_set"
    SLOT_WAS_NOT_SET = "slot_was_not_set"
    BOT_UTTERED = "bot_uttered"
    BOT_DID_NOT_UTTER = "bot_did_not_utter"
    GENERATIVE_RESPONSE_IS_RELEVANT = "generative_response_is_relevant"
    GENERATIVE_RESPONSE_IS_GROUNDED = "generative_response_is_grounded"


class AssertionOperator(Enum):
    ANY = "any"
    ALL = "all"


@lru_cache(maxsize=1)
def _get_all_assertion_subclasses() -> Dict[str, Type[Assertion]]:
    return {
        sub_class.type(): sub_class
        for sub_class in rasa.shared.utils.common.all_subclasses(Assertion)
    }


class InvalidAssertionType(RasaException):
    """Raised if an assertion type is invalid."""

    def __init__(self, assertion_type: str) -> None:
        """Creates a `InvalidAssertionType`.

        Args:
            assertion_type: The invalid assertion type.
        """
        super().__init__(f"Invalid assertion type '{assertion_type}'.")


@dataclass
class Assertion:
    """Base class for storing assertions."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the assertion."""
        raise NotImplementedError

    @staticmethod
    def from_dict(assertion_dict: Dict[Text, Any]) -> Assertion:
        """Creates an assertion from a dictionary."""
        raise NotImplementedError

    def as_dict(self) -> Dict[str, Any]:
        """Return the `Assertion` as a dictionary.

        Returns:
            The `Assertion` as a dictionary.
        """
        data = dataclasses.asdict(self)
        data["type"] = self.type()
        if OPERATOR_KEY in data and data[OPERATOR_KEY] is not None:
            data[OPERATOR_KEY] = data[OPERATOR_KEY].value
        return data

    @staticmethod
    def create_typed_assertion(data: Dict[str, Any]) -> Assertion:
        """Creates a `Assertion` from a dictionary.

        Args:
            data: The dictionary to create the `Assertion` from.

        Returns:
            The created `Assertion`.
        """
        typ = next(iter(data.keys()))

        subclass_mapping = _get_all_assertion_subclasses()

        clazz = subclass_mapping.get(typ)

        if clazz is None:
            structlogger.warning("assertion.unknown_type", data=data)
            raise InvalidAssertionType(typ)

        try:
            return clazz.from_dict(data)
        except NotImplementedError:
            structlogger.warning("assertion.unknown_type", data=data)
            raise InvalidAssertionType(typ)

    def run(
        self,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
        **kwargs: Any,
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the assertion on the given events for that user turn.

        Args:
            turn_events: The events to run the assertion on.
            prior_events: All events prior to the current turn.
            assertion_order_error_message: The error message to append if the assertion
                order is enabled.
            kwargs: Additional keyword arguments.

        Returns:
            A tuple of the assertion failure and the matching event if the assertion
            passes, otherwise `None`.
        """
        raise NotImplementedError

    def generate_test_failure(
        self,
        error_message: str,
        assertion_order_error_message: str,
        prior_events: List[Event],
        turn_events: List[Event],
        line: Optional[int] = None,
    ) -> Tuple[AssertionFailure, None]:
        """Generate an assertion failure with the given error message."""
        error_message += assertion_order_error_message

        return self._generate_assertion_failure(
            error_message, prior_events, turn_events, line
        )

    def _generate_assertion_failure(
        self,
        error_message: str,
        prior_events: List[Event],
        turn_events: List[Event],
        line: Optional[int] = None,
    ) -> Tuple[AssertionFailure, None]:
        return AssertionFailure(
            assertion=self,
            error_message=error_message,
            actual_events_transcript=create_actual_events_transcript(
                prior_events, turn_events
            ),
            error_line=line,
        ), None


@dataclass
class FlowStartedAssertion(Assertion):
    """Class for storing the flow started assertion."""

    flow_id: Optional[str] = None
    line: Optional[int] = None
    operator: Optional[AssertionOperator] = None
    flow_ids: List[str] = dataclasses.field(default_factory=list)

    @classmethod
    def type(cls) -> str:
        return AssertionType.FLOW_STARTED.value

    @staticmethod
    def from_dict(assertion_dict: Dict[Text, Any]) -> FlowStartedAssertion:
        assertion_value = assertion_dict.get(AssertionType.FLOW_STARTED.value)
        operator = (
            assertion_value.get(OPERATOR_KEY, "").lower()
            if isinstance(assertion_value, dict)
            else None
        )
        flow_ids = (
            assertion_value.get(FLOW_IDS_KEY, [])
            if isinstance(assertion_value, dict)
            else []
        )
        return FlowStartedAssertion(
            flow_id=assertion_value if isinstance(assertion_value, str) else None,
            line=assertion_dict.lc.line + 1 if hasattr(assertion_dict, "lc") else None,
            operator=AssertionOperator(operator) if operator else None,
            flow_ids=flow_ids,
        )

    def run(
        self,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
        **kwargs: Any,
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the flow started assertion on the given events for that user turn."""
        if self.flow_id is not None:
            rasa.shared.utils.io.raise_deprecation_warning(
                f"'{AssertionType.FLOW_STARTED.value}' assertions defining "
                f"a single 'flow_id' value are "
                f"deprecated and will be removed in a future Rasa version. "
                f"Please use the '{FLOW_IDS_KEY}' field with an "
                f"'{OPERATOR_KEY}' instead.",
            )
            return self._run_single_flow_id_check(
                self.flow_id,
                turn_events,
                prior_events,
                assertion_order_error_message,
            )

        if self.operator == AssertionOperator.ALL:
            flow_id = next(
                iter(self.flow_ids)
            )  # there should be exactly one flow ID when using 'all' operator
            return self._run_single_flow_id_check(
                flow_id,
                turn_events,
                prior_events,
                assertion_order_error_message,
            )

        if self.operator == AssertionOperator.ANY:
            return self._run_any_flow_id_check(
                turn_events,
                prior_events,
                assertion_order_error_message,
            )

        raise RasaException(
            f"Invalid operator '{self.operator}' for 'flow_started' assertion. "
            "Supported operators are 'any' and 'all'."
        )

    def _run_single_flow_id_check(
        self,
        flow_id: str,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the deprecated flow_id check for backward compatibility."""
        try:
            matching_event = next(
                event
                for event in turn_events
                if isinstance(event, FlowStarted) and event.flow_id == flow_id
            )
        except StopIteration:
            error_message = f"Flow with id '{flow_id}' did not start."
            return self.generate_test_failure(
                error_message,
                assertion_order_error_message,
                prior_events,
                turn_events,
                self.line,
            )

        return None, matching_event

    def _run_any_flow_id_check(
        self,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the 'any' operator flow_id check."""
        for flow_id in self.flow_ids:
            try:
                matching_event = next(
                    event
                    for event in turn_events
                    if isinstance(event, FlowStarted) and event.flow_id == flow_id
                )
                return None, matching_event
            except StopIteration:
                continue

        error_message = (
            f"None of the flows with ids '{', '.join(self.flow_ids)}' started."
        )
        return self.generate_test_failure(
            error_message,
            assertion_order_error_message,
            prior_events,
            turn_events,
            self.line,
        )

    def __hash__(self) -> int:
        return hash(json.dumps(self.as_dict()))


@dataclass
class FlowCompletedAssertion(Assertion):
    """Class for storing the flow completed assertion."""

    flow_id: str
    flow_step_id: Optional[str] = None
    line: Optional[int] = None

    @classmethod
    def type(cls) -> str:
        return AssertionType.FLOW_COMPLETED.value

    @staticmethod
    def from_dict(assertion_dict: Dict[Text, Any]) -> FlowCompletedAssertion:
        line = assertion_dict.lc.line + 1 if hasattr(assertion_dict, "lc") else None
        assertion_dict = assertion_dict.get(AssertionType.FLOW_COMPLETED.value, {})

        return FlowCompletedAssertion(
            flow_id=assertion_dict.get("flow_id"),
            flow_step_id=assertion_dict.get("flow_step_id"),
            line=line,
        )

    def run(
        self,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
        **kwargs: Any,
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the flow completed assertion on the given events for that user turn."""
        try:
            matching_event = next(
                event
                for event in turn_events
                if isinstance(event, FlowCompleted) and event.flow_id == self.flow_id
            )
        except StopIteration:
            error_message = f"Flow with id '{self.flow_id}' did not complete."
            return self.generate_test_failure(
                error_message,
                assertion_order_error_message,
                prior_events,
                turn_events,
                self.line,
            )

        if (
            self.flow_step_id is not None
            and matching_event.step_id != self.flow_step_id
        ):
            error_message = (
                f"Flow with id '{self.flow_id}' did not complete "
                f"at expected step id '{self.flow_step_id}'. The actual "
                f"step id was '{matching_event.step_id}'."
            )
            return self.generate_test_failure(
                error_message,
                assertion_order_error_message,
                prior_events,
                turn_events,
                self.line,
            )

        return None, matching_event

    def __hash__(self) -> int:
        return hash(json.dumps(self.as_dict()))


@dataclass
class FlowCancelledAssertion(Assertion):
    """Class for storing the flow cancelled assertion."""

    flow_id: str
    flow_step_id: Optional[str] = None
    line: Optional[int] = None

    @classmethod
    def type(cls) -> str:
        return AssertionType.FLOW_CANCELLED.value

    @staticmethod
    def from_dict(assertion_dict: Dict[Text, Any]) -> FlowCancelledAssertion:
        line = assertion_dict.lc.line + 1 if hasattr(assertion_dict, "lc") else None
        assertion_dict = assertion_dict.get(AssertionType.FLOW_CANCELLED.value, {})

        return FlowCancelledAssertion(
            flow_id=assertion_dict.get("flow_id"),
            flow_step_id=assertion_dict.get("flow_step_id"),
            line=line,
        )

    def run(
        self,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
        **kwargs: Any,
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the flow cancelled assertion on the given events for that user turn."""
        try:
            matching_event = next(
                event
                for event in turn_events
                if isinstance(event, FlowCancelled) and event.flow_id == self.flow_id
            )
        except StopIteration:
            error_message = f"Flow with id '{self.flow_id}' was not cancelled."
            return self.generate_test_failure(
                error_message,
                assertion_order_error_message,
                prior_events,
                turn_events,
                self.line,
            )

        if (
            self.flow_step_id is not None
            and matching_event.step_id != self.flow_step_id
        ):
            error_message = (
                f"Flow with id '{self.flow_id}' was not cancelled "
                f"at expected step id '{self.flow_step_id}'. The actual "
                f"step id was '{matching_event.step_id}'."
            )
            return self.generate_test_failure(
                error_message,
                assertion_order_error_message,
                prior_events,
                turn_events,
                self.line,
            )

        return None, matching_event

    def __hash__(self) -> int:
        return hash(json.dumps(self.as_dict()))


@dataclass
class PatternClarificationContainsAssertion(Assertion):
    """Class for storing the pattern clarification contains assertion."""

    flow_names: Set[str]
    line: Optional[int] = None
    operator: Optional[AssertionOperator] = None
    flow_ids: List[str] = dataclasses.field(default_factory=list)

    @classmethod
    def type(cls) -> str:
        return AssertionType.PATTERN_CLARIFICATION_CONTAINS.value

    @staticmethod
    def from_dict(
        assertion_dict: Dict[Text, Any],
    ) -> PatternClarificationContainsAssertion:
        assertion_value = assertion_dict.get(
            AssertionType.PATTERN_CLARIFICATION_CONTAINS.value
        )

        operator = (
            assertion_value.get(OPERATOR_KEY, "").lower()
            if isinstance(assertion_value, dict)
            else None
        )

        flow_ids = (
            assertion_value.get("flow_ids", [])
            if isinstance(assertion_value, dict)
            else []
        )

        return PatternClarificationContainsAssertion(
            flow_names=set(assertion_value)
            if isinstance(assertion_value, list)
            else set(),
            line=assertion_dict.lc.line + 1 if hasattr(assertion_dict, "lc") else None,
            operator=AssertionOperator(operator) if operator else None,
            flow_ids=flow_ids,
        )

    def run(
        self,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
        **kwargs: Any,
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the flow completed assertion on the given events for that user turn."""
        try:
            matching_event = next(
                event
                for event in turn_events
                if isinstance(event, FlowStarted)
                and event.flow_id == FLOW_PATTERN_CLARIFICATION
            )
        except StopIteration:
            error_message = f"'{FLOW_PATTERN_CLARIFICATION}' pattern did not trigger."
            return self.generate_test_failure(
                error_message,
                assertion_order_error_message,
                prior_events,
                turn_events,
                self.line,
            )

        if self.operator is None and self.flow_names:
            return self._run_flow_names_check(
                matching_event,
                turn_events,
                prior_events,
                assertion_order_error_message,
            )

        actual_flow_ids = set(matching_event.metadata.get("clarification_ids", []))

        if self.operator == AssertionOperator.ALL:
            return self._run_all_flow_id_check(
                actual_flow_ids,
                matching_event,
                turn_events,
                prior_events,
                assertion_order_error_message,
            )

        if self.operator == AssertionOperator.ANY:
            return self._run_any_flow_id_check(
                actual_flow_ids,
                matching_event,
                turn_events,
                prior_events,
                assertion_order_error_message,
            )

        raise RasaException(
            f"Invalid operator '{self.operator}' for "
            f"'{FLOW_PATTERN_CLARIFICATION}' assertion. "
            "Supported operators are 'any' and 'all'."
        )

    def _run_flow_names_check(
        self,
        matching_event: Event,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the flow names check."""
        rasa.shared.utils.io.raise_deprecation_warning(
            f"'{AssertionType.PATTERN_CLARIFICATION_CONTAINS.value}' "
            f"assertions defining a list of flow names are deprecated "
            f"and will be removed in a future Rasa version. "
            f"Please use the '{FLOW_IDS_KEY}' field with an '{OPERATOR_KEY}' instead.",
        )

        actual_flow_names = set(matching_event.metadata.get("names", set()))
        if actual_flow_names != self.flow_names:
            error_message = (
                f"'{FLOW_PATTERN_CLARIFICATION}' pattern did not contain "
                f"the expected options. Expected options: {self.flow_names}. "
            )
            return self.generate_test_failure(
                error_message,
                assertion_order_error_message,
                prior_events,
                turn_events,
                self.line,
            )

        return None, matching_event

    def _run_all_flow_id_check(
        self,
        actual_flow_ids: Set[str],
        matching_event: Event,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        if actual_flow_ids != set(self.flow_ids):
            error_message = (
                f"'{FLOW_PATTERN_CLARIFICATION}' pattern did not contain all of "
                f"the expected options '{', '.join(self.flow_ids)}'. "
                f"Actual options: '{', '.join(actual_flow_ids)}'."
            )
            return self.generate_test_failure(
                error_message,
                assertion_order_error_message,
                prior_events,
                turn_events,
                self.line,
            )

        return None, matching_event

    def _run_any_flow_id_check(
        self,
        actual_flow_ids: Set[str],
        matching_event: Event,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the 'any' operator flow_id check."""
        for flow_id in self.flow_ids:
            if flow_id in actual_flow_ids:
                return None, matching_event

        error_message = (
            f"'{FLOW_PATTERN_CLARIFICATION}' pattern did not contain "
            f"any of the expected options '{', '.join(self.flow_ids)}'."
        )
        return self.generate_test_failure(
            error_message,
            assertion_order_error_message,
            prior_events,
            turn_events,
            self.line,
        )

    def __hash__(self) -> int:
        return hash(json.dumps(self.as_dict(), cls=SetEncoder))


@dataclass
class ActionExecutedAssertion(Assertion):
    """Class for storing the action executed assertion."""

    action_name: str
    line: Optional[int] = None

    @classmethod
    def type(cls) -> str:
        return AssertionType.ACTION_EXECUTED.value

    @staticmethod
    def from_dict(assertion_dict: Dict[Text, Any]) -> ActionExecutedAssertion:
        return ActionExecutedAssertion(
            action_name=assertion_dict.get(AssertionType.ACTION_EXECUTED.value),
            line=assertion_dict.lc.line + 1 if hasattr(assertion_dict, "lc") else None,
        )

    def run(
        self,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
        **kwargs: Any,
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the action executed assertion on the given events for that user turn."""
        step_index = kwargs.get("step_index")
        original_turn_events, turn_events = _get_turn_events_based_on_step_index(
            step_index, turn_events, prior_events
        )

        try:
            matching_event = next(
                event
                for event in turn_events
                if isinstance(event, ActionExecuted)
                and event.action_name == self.action_name
            )
        except StopIteration:
            error_message = f"Action '{self.action_name}' did not execute."
            return self.generate_test_failure(
                error_message,
                assertion_order_error_message,
                prior_events,
                original_turn_events,
                self.line,
            )

        return None, matching_event

    def __hash__(self) -> int:
        return hash(json.dumps(self.as_dict()))


@dataclass
class AssertedSlot:
    """Class for storing information asserted about slots."""

    name: str
    value: Any
    line: Optional[int] = None

    @staticmethod
    def from_dict(slot_dict: Dict[Text, Any]) -> AssertedSlot:
        return AssertedSlot(
            name=slot_dict.get("name"),
            value=slot_dict.get("value", "value key is undefined"),
            line=slot_dict.lc.line + 1 if hasattr(slot_dict, "lc") else None,
        )


@dataclass
class SlotWasSetAssertion(Assertion):
    """Class for storing the slot was set assertion."""

    slots: List[AssertedSlot]

    @classmethod
    def type(cls) -> str:
        return AssertionType.SLOT_WAS_SET.value

    @staticmethod
    def from_dict(assertion_dict: Dict[Text, Any]) -> SlotWasSetAssertion:
        return SlotWasSetAssertion(
            slots=[
                AssertedSlot.from_dict(slot)
                for slot in assertion_dict.get(AssertionType.SLOT_WAS_SET.value, [])
            ],
        )

    def run(
        self,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
        **kwargs: Any,
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the slot_was_set assertion on the given events for that user turn."""
        matching_event = None

        step_index = kwargs.get("step_index")
        original_turn_events, turn_events = _get_turn_events_based_on_step_index(
            step_index, turn_events, prior_events
        )

        for slot in self.slots:
            matching_events = [
                event
                for event in turn_events
                if isinstance(event, SlotSet) and event.key == slot.name
            ]
            if not matching_events:
                error_message = f"Slot '{slot.name}' was not set."
                return self.generate_test_failure(
                    error_message,
                    assertion_order_error_message,
                    prior_events,
                    original_turn_events,
                    slot.line,
                )

            if slot.value == "value key is undefined":
                matching_event = matching_events[0]
                structlogger.debug(
                    "slot_was_set_assertion.run",
                    last_event_seen=matching_event,
                    event_info="Slot value is not asserted and we have "
                    "multiple events for the same slot. "
                    "We will mark the first event as last event seen.",
                )
                continue

            try:
                matching_event = next(
                    event for event in matching_events if event.value == slot.value
                )
            except StopIteration:
                error_message = (
                    f"Slot '{slot.name}' was set to a different value "
                    f"'{matching_events[-1].value}' than the "
                    f"expected '{slot.value}' value."
                )
                return self.generate_test_failure(
                    error_message,
                    assertion_order_error_message,
                    prior_events,
                    original_turn_events,
                    slot.line,
                )

        return None, matching_event

    def __hash__(self) -> int:
        return hash(json.dumps(self.as_dict()))


@dataclass
class SlotWasNotSetAssertion(Assertion):
    """Class for storing the slot was not set assertion."""

    slots: List[AssertedSlot]

    @classmethod
    def type(cls) -> str:
        return AssertionType.SLOT_WAS_NOT_SET.value

    @staticmethod
    def from_dict(assertion_dict: Dict[Text, Any]) -> SlotWasNotSetAssertion:
        return SlotWasNotSetAssertion(
            slots=[
                AssertedSlot.from_dict(slot)
                for slot in assertion_dict.get(AssertionType.SLOT_WAS_NOT_SET.value, [])
            ]
        )

    def run(
        self,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
        **kwargs: Any,
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the slot_was_not_set assertion on the given events for that user turn."""
        matching_event = None

        step_index = kwargs.get("step_index")
        original_turn_events, turn_events = _get_turn_events_based_on_step_index(
            step_index, turn_events, prior_events
        )

        for slot in self.slots:
            matching_events = [
                event
                for event in turn_events
                if isinstance(event, SlotSet) and event.key == slot.name
            ]
            if not matching_events:
                continue

            # take the most recent event in the list of matching events
            # since that is the final value in the tracker for that user turn
            matching_event = matching_events[-1]

            if (
                slot.value == "value key is undefined"
                and matching_event.value is not None
            ):
                error_message = (
                    f"Slot '{slot.name}' was set to '{matching_event.value}' but "
                    f"it should not have been set."
                )
                return self.generate_test_failure(
                    error_message,
                    assertion_order_error_message,
                    prior_events,
                    turn_events,
                    slot.line,
                )

            if matching_event.value == slot.value:
                error_message = (
                    f"Slot '{slot.name}' was set to '{slot.value}' "
                    f"but it should not have been set."
                )
                return self.generate_test_failure(
                    error_message,
                    assertion_order_error_message,
                    prior_events,
                    original_turn_events,
                    slot.line,
                )

        return None, matching_event

    def __hash__(self) -> int:
        return hash(json.dumps(self.as_dict()))


@dataclass
class AssertedButton:
    """Class for storing information asserted about buttons."""

    title: str
    payload: Optional[str] = None

    @staticmethod
    def from_dict(button_dict: Dict[Text, Any]) -> AssertedButton:
        return AssertedButton(
            title=button_dict.get("title"),
            payload=button_dict.get("payload"),
        )


@dataclass
class BotUtteredAssertion(Assertion):
    """Class for storing the bot uttered assertion."""

    utter_name: Optional[str] = None
    text_matches: Optional[str] = None
    buttons: Optional[List[AssertedButton]] = None
    line: Optional[int] = None

    @classmethod
    def type(cls) -> str:
        return AssertionType.BOT_UTTERED.value

    @staticmethod
    def from_dict(assertion_dict: Dict[Text, Any]) -> BotUtteredAssertion:
        utter_name, text_matches, buttons = (
            BotUtteredAssertion._extract_assertion_properties(assertion_dict)
        )

        if BotUtteredAssertion._assertion_is_empty(utter_name, text_matches, buttons):
            raise RasaException(
                "A 'bot_uttered' assertion is empty, it should contain at least one "
                "of the allowed properties: 'utter_name', 'text_matches', 'buttons'."
            )

        return BotUtteredAssertion(
            utter_name=utter_name,
            text_matches=text_matches,
            buttons=buttons,
            line=assertion_dict.lc.line + 1 if hasattr(assertion_dict, "lc") else None,
        )

    @staticmethod
    def _extract_assertion_properties(
        assertion_dict: Dict[Text, Any],
    ) -> Tuple[Optional[str], Optional[str], List[AssertedButton]]:
        """Extracts the assertion properties from a dictionary."""
        assertion_dict = assertion_dict.get(AssertionType.BOT_UTTERED.value, {})
        utter_name = assertion_dict.get("utter_name")
        text_matches = assertion_dict.get("text_matches")
        buttons = [
            AssertedButton.from_dict(button)
            for button in assertion_dict.get("buttons", [])
        ]

        return utter_name, text_matches, buttons

    @staticmethod
    def _assertion_is_empty(
        utter_name: Optional[str],
        text_matches: Optional[str],
        buttons: List[AssertedButton],
    ) -> bool:
        """Validate if the bot uttered assertion is empty."""
        if not utter_name and not text_matches and not buttons:
            return True

        return False

    def run(
        self,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
        **kwargs: Any,
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the bot_uttered assertion on the given events for that user turn."""
        matching_event = None
        error_messages = []

        step_index = kwargs.get("step_index")
        original_turn_events, turn_events = _get_turn_events_based_on_step_index(
            step_index, turn_events, prior_events
        )

        if self.utter_name is not None:
            try:
                matching_event = next(
                    event
                    for event in turn_events
                    if isinstance(event, BotUttered)
                    and event.metadata.get("utter_action") == self.utter_name
                )
            except StopIteration:
                error_messages.append(
                    f"Bot did not utter '{self.utter_name}' response."
                )

        if self.text_matches is not None:
            pattern = re.compile(self.text_matches)
            try:
                matching_event = next(
                    event
                    for event in turn_events
                    if isinstance(event, BotUttered) and pattern.search(event.text)
                )
            except StopIteration:
                error_messages.append(
                    f"Bot did not utter any response which "
                    f"matches the provided text pattern "
                    f"'{self.text_matches}'."
                )

        if self.buttons:
            try:
                matching_event = next(
                    event
                    for event in turn_events
                    if isinstance(event, BotUttered) and self._buttons_match(event)
                )
            except StopIteration:
                error_messages.append(
                    "Bot did not utter any response with the expected buttons."
                )

        if error_messages:
            error_message = " ".join(error_messages)
            return self.generate_test_failure(
                error_message,
                assertion_order_error_message,
                prior_events,
                original_turn_events,
                self.line,
            )
        return None, matching_event

    def _buttons_match(self, event: BotUttered) -> bool:
        """Check if the bot response contains the expected buttons."""
        # a button is a dictionary with keys 'title' and 'payload'
        actual_buttons = event.data.get("buttons", [])
        if not actual_buttons:
            return False

        return all(
            self._button_matches(actual_button, expected_button)
            for actual_button, expected_button in zip(actual_buttons, self.buttons)
        )

    @staticmethod
    def _button_matches(
        actual_button: Dict[str, Any], expected_button: AssertedButton
    ) -> bool:
        """Check if the actual button matches the expected button."""
        return (
            actual_button.get("title") == expected_button.title
            and actual_button.get("payload") == expected_button.payload
        )

    def __hash__(self) -> int:
        return hash(json.dumps(self.as_dict()))


@dataclass
class BotDidNotUtterAssertion(Assertion):
    """Class for the 'bot_did_not_utter' assertion."""

    utter_name: Optional[str] = None
    text_matches: Optional[str] = None
    buttons: Optional[List[AssertedButton]] = None
    line: Optional[int] = None

    @classmethod
    def type(cls) -> str:
        return AssertionType.BOT_DID_NOT_UTTER.value

    @staticmethod
    def from_dict(assertion_dict: Dict[Text, Any]) -> BotDidNotUtterAssertion:
        """Creates a BotDidNotUtterAssertion from a dictionary."""
        assertion_dict = assertion_dict.get(AssertionType.BOT_DID_NOT_UTTER.value, {})
        utter_name = assertion_dict.get("utter_name")
        text_matches = assertion_dict.get("text_matches")
        buttons = [
            AssertedButton.from_dict(button)
            for button in assertion_dict.get("buttons", [])
        ]

        if not utter_name and not text_matches and not buttons:
            raise RasaException(
                "A 'bot_did_not_utter' assertion is empty. "
                "It should contain at least one of the allowed properties: "
                "'utter_name', 'text_matches', or 'buttons'."
            )

        return BotDidNotUtterAssertion(
            utter_name=utter_name,
            text_matches=text_matches,
            buttons=buttons,
            line=assertion_dict.lc.line + 1 if hasattr(assertion_dict, "lc") else None,
        )

    def run(
        self,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
        **kwargs: Any,
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Checks that the bot did not utter the specified messages or buttons."""
        step_index = kwargs.get("step_index")
        original_turn_events, turn_events = _get_turn_events_based_on_step_index(
            step_index, turn_events, prior_events
        )

        for event in turn_events:
            if isinstance(event, BotUttered):
                error_messages = []
                if self._utter_name_matches(event):
                    error_messages.append(
                        f"Bot uttered a forbidden utterance '{self.utter_name}'."
                    )
                if self._text_matches(event):
                    error_messages.append(
                        f"Bot uttered a forbidden message matching "
                        f"the pattern '{self.text_matches}'."
                    )
                if self._buttons_match(event):
                    error_messages.append(
                        "Bot uttered a forbidden response with specified buttons."
                    )

                if error_messages:
                    error_message = " ".join(error_messages)
                    return self.generate_test_failure(
                        error_message,
                        assertion_order_error_message,
                        prior_events,
                        original_turn_events,
                        self.line,
                    )
        return None, None

    def _utter_name_matches(self, event: BotUttered) -> bool:
        if self.utter_name is not None:
            if event.metadata.get("utter_action") == self.utter_name:
                return True
        return False

    def _text_matches(self, event: BotUttered) -> bool:
        if self.text_matches is not None:
            pattern = re.compile(self.text_matches)
            if pattern.search(event.text):
                return True
        return False

    def _buttons_match(self, event: BotUttered) -> bool:
        """Check if the bot response contains any of the forbidden buttons."""
        if self.buttons is None:
            return False

        actual_buttons = event.data.get("buttons", [])
        if not actual_buttons:
            return False

        for actual_button in actual_buttons:
            if any(
                self._is_forbidden_button(actual_button, forbidden_button)
                for forbidden_button in self.buttons
            ):
                return True
        return False

    @staticmethod
    def _is_forbidden_button(
        actual_button: Dict[str, Any], forbidden_button: AssertedButton
    ) -> bool:
        """Check if the button matches any of the forbidden buttons."""
        actual_title = actual_button.get("title")
        actual_payload = actual_button.get("payload")

        title_matches = forbidden_button.title == actual_title
        payload_matches = forbidden_button.payload == actual_payload
        if title_matches and payload_matches:
            return True
        return False

    def __hash__(self) -> int:
        """Hash method to ensure the assertion is hashable."""
        return hash(json.dumps(self.as_dict()))


@dataclass
class GenerativeResponseMixin(Assertion):
    """Mixin class for storing generative response assertions."""

    metric_adjective: str
    threshold: float = DEFAULT_THRESHOLD
    utter_name: Optional[str] = None
    utter_source: Optional[str] = None
    line: Optional[int] = None

    @classmethod
    def type(cls) -> str:
        return ""

    def as_dict(self) -> Dict[str, Any]:
        data = super().as_dict()
        data.pop("metric_adjective")
        return data

    def _render_prompt(self, matching_event: BotUttered) -> str:
        raise NotImplementedError

    def _get_processed_output(self, parsed_llm_output: Dict[str, Any]) -> List[Any]:
        raise NotImplementedError

    def _process_response(
        self, llm_response: str, bot_message: str
    ) -> List[Dict[str, Any]]:
        """Process the LLM response."""
        parsed_llm_output = _parse_llm_output(llm_response, bot_message)
        _validate_parsed_llm_output(parsed_llm_output, bot_message)

        processed_output = self._get_processed_output(parsed_llm_output)
        return processed_output

    def _run_llm_evaluation(
        self,
        matching_event: BotUttered,
        step_text: str,
        llm_judge_config: "LLMJudgeConfig",
        assertion_order_error_message: str,
        prior_events: List[Event],
        turn_events: List[Event],
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the LLM evaluation on the given event."""
        bot_message = matching_event.text
        prompt = self._render_prompt(matching_event)
        llm_response = self._invoke_llm(llm_judge_config, prompt)

        try:
            processed_output = self._process_response(llm_response, bot_message)
        except RasaException as exc:
            structlogger.error(
                "e2e_test.generative_response_evaluation.error", error=exc
            )
            return self._generate_assertion_failure(
                str(exc), prior_events, turn_events, self.line
            )

        score_inputs = ScoreInputs(
            threshold=self.threshold,
            matching_event=matching_event,
            user_question=step_text,
            llm_judge_config=llm_judge_config,
        )
        score, error_justification = calculate_score(
            assertion_type=self.type(),
            processed_output=processed_output,
            score_inputs=score_inputs,
        )

        if score < self.threshold:
            error_message = (
                f"Generative response '{matching_event.text}' "
                f"given to the user input '{step_text}' "
                f"was not {self.metric_adjective}. "
                f"Expected score to be above '{self.threshold}' threshold, "
                f"but was '{round(score,2)}'. The LLM Judge model has justified its "
                f"score like so: {error_justification}."
            )
            return self.generate_test_failure(
                error_message,
                assertion_order_error_message,
                prior_events,
                turn_events,
                self.line,
            )

        return None, matching_event

    def _invoke_llm(self, llm_judge_config: LLMJudgeConfig, prompt: str) -> str:
        """Invoke the LLM to evaluate the generative response."""
        structlogger.debug(
            f"generative_response_is_{self.metric_adjective}_assertion.run_llm_evaluation",
        )

        llm = llm_factory(
            llm_judge_config.llm_config_as_dict,
            llm_judge_config.get_default_llm_config(),
        )

        try:
            llm_response = llm.completion(prompt)
            return llm_response.choices[0]
        except Exception as exc:
            structlogger.error(
                "e2e_test.generative_response_evaluation.llm.error", error=exc
            )
            raise ProviderClientAPIException(
                message="LLM call exception", original_exception=exc
            )

    def _run_assertion_with_utter_name(
        self,
        matching_events: List[BotUttered],
        step_text: str,
        llm_judge_config: "LLMJudgeConfig",
        assertion_order_error_message: str,
        prior_events: List[Event],
        turn_events: List[Event],
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Assert metric for the given utter name."""
        try:
            matching_event = next(
                event
                for event in matching_events
                if event.metadata.get("utter_action") == self.utter_name
            )
        except StopIteration:
            error_message = f"Bot did not utter '{self.utter_name}' response."
            return self.generate_test_failure(
                error_message,
                assertion_order_error_message,
                prior_events,
                turn_events,
                self.line,
            )

        return self._run_llm_evaluation(
            matching_event,
            step_text,
            llm_judge_config,
            assertion_order_error_message,
            prior_events,
            turn_events,
        )

    def run(
        self,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
        llm_judge_config: Optional["LLMJudgeConfig"] = None,
        step_text: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the LLM evaluation on the given events for that user turn."""
        matching_events: List[BotUttered] = _find_matching_generative_events(
            turn_events, self.utter_source
        )

        if not matching_events:
            error_message = (
                "No generative response issued by either the Enterprise Search Policy, "
                "IntentlessPolicy, Contextual Response Rephraser or a "
                "named custom action was found, "
                "but one was expected. Please define an utter_source in the assertion "
                "to specify which generative response to evaluate."
            )
            error_message += assertion_order_error_message

            return self._generate_assertion_failure(
                error_message, prior_events, turn_events, self.line
            )

        if self.utter_name is not None:
            return self._run_assertion_with_utter_name(
                matching_events,
                step_text,
                llm_judge_config,
                assertion_order_error_message,
                prior_events,
                turn_events,
            )

        if len(matching_events) > 1:
            structlogger.debug(
                f"generative_response_is_{self.metric_adjective}_assertion.run",
                event_info=f"Multiple generative responses found, "
                f"we will evaluate the first of the responses "
                f"'{matching_events[0].text}'.",
            )

        matching_event = matching_events[0]

        return self._run_llm_evaluation(
            matching_event,
            step_text,
            llm_judge_config,
            assertion_order_error_message,
            prior_events,
            turn_events,
        )


@dataclass
class GenerativeResponseIsRelevantAssertion(GenerativeResponseMixin):
    """Class for storing the generative response is relevant assertion."""

    @classmethod
    def type(cls) -> str:
        return AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value

    def _render_prompt(self, matching_event: BotUttered) -> str:
        """Render the prompt."""
        inputs = _get_prompt_inputs(self.type(), matching_event)
        prompt_template = _get_default_prompt_template(
            DEFAULT_ANSWER_RELEVANCE_PROMPT_TEMPLATE_FILE_NAME
        )
        return Template(prompt_template).render(**inputs)

    @staticmethod
    def from_dict(
        assertion_dict: Dict[Text, Any],
    ) -> GenerativeResponseIsRelevantAssertion:
        assertion_dict = assertion_dict.get(
            AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value, {}
        )

        return GenerativeResponseIsRelevantAssertion(
            threshold=assertion_dict.get(KEY_THRESHOLD, DEFAULT_THRESHOLD),
            utter_name=assertion_dict.get(KEY_UTTER_NAME),
            line=assertion_dict.lc.line + 1 if hasattr(assertion_dict, "lc") else None,
            metric_adjective="relevant",
            utter_source=assertion_dict.get(KEY_UTTER_SOURCE),
        )

    def __hash__(self) -> int:
        return hash(json.dumps(self.as_dict()))

    def _get_processed_output(self, parsed_llm_output: Dict[str, Any]) -> List[Any]:
        questions = parsed_llm_output.get("question_variations", [])
        if not questions:
            raise RasaException(
                "No question variations were extracted by the LLM Judge."
            )
        return questions


@dataclass
class GenerativeResponseIsGroundedAssertion(GenerativeResponseMixin):
    """Class for storing the generative response is grounded assertion."""

    ground_truth: Optional[str] = None

    @classmethod
    def type(cls) -> str:
        return AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value

    def _render_prompt(self, matching_event: BotUttered) -> str:
        """Render the prompt."""
        inputs = _get_prompt_inputs(
            assertion_type=self.type(),
            matching_event=matching_event,
            ground_truth=self.ground_truth,
        )
        prompt_template = _get_default_prompt_template(
            DEFAULT_GROUNDEDNESS_PROMPT_TEMPLATE_FILE_NAME
        )
        return Template(prompt_template).render(**inputs)

    @staticmethod
    def from_dict(
        assertion_dict: Dict[Text, Any],
    ) -> GenerativeResponseIsGroundedAssertion:
        assertion_dict = assertion_dict.get(
            AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value, {}
        )

        return GenerativeResponseIsGroundedAssertion(
            threshold=assertion_dict.get(KEY_THRESHOLD, DEFAULT_THRESHOLD),
            utter_name=assertion_dict.get(KEY_UTTER_NAME),
            ground_truth=assertion_dict.get(KEY_GROUND_TRUTH),
            line=assertion_dict.lc.line + 1 if hasattr(assertion_dict, "lc") else None,
            metric_adjective="grounded",
            utter_source=assertion_dict.get(KEY_UTTER_SOURCE),
        )

    def __hash__(self) -> int:
        return hash(json.dumps(self.as_dict()))

    def _get_processed_output(self, parsed_llm_output: Dict[str, Any]) -> List[Any]:
        """Process the LLM response."""
        statements = parsed_llm_output.get("statements", [])
        if not statements:
            raise RasaException(
                "No statements were extracted and scored by the LLM Judge. "
                "Please check the LLM Judge configuration"
            )

        return statements


@dataclass
class AssertionFailure:
    """Class for storing the assertion failure."""

    assertion: Assertion
    error_message: Text
    actual_events_transcript: List[Text]
    error_line: Optional[int] = None

    def as_dict(self) -> Dict[Text, Any]:
        """Returns the assertion failure as a dictionary."""
        return {
            "assertion": self.assertion.as_dict(),
            "error_message": self.error_message,
            "actual_events_transcript": self.actual_events_transcript,
        }


def create_actual_events_transcript(
    prior_events: List[Event], turn_events: List[Event]
) -> List[Text]:
    """Create the actual events transcript for the assertion failure."""
    all_events = prior_events + turn_events

    event_transcript = []

    for event in all_events:
        if isinstance(event, SlotSet) and event.key in DEFAULT_SLOT_NAMES:
            continue
        if isinstance(event, DefinePrevUserUtteredFeaturization):
            continue
        if isinstance(event, DialogueStackUpdated):
            continue

        event_transcript.append(repr(event))

    return event_transcript


def _get_turn_events_based_on_step_index(
    step_index: int, turn_events: List[Event], prior_events: List[Event]
) -> Tuple[List[Event], List[Event]]:
    """Get the turn events based on the step index.

    For the first step, we need to include the prior events as well
    in the same user turn. For the subsequent steps, we only need the
    events that follow the user uttered event on which the tracker
    was originally sliced by.

    Returns:
        List[Event]: The copy of turn_events
        List[Event]: The turn events based on the step index

    """
    original_turn_events = turn_events[:]
    if step_index == 0:
        return original_turn_events, prior_events + turn_events

    return original_turn_events, turn_events


def _get_default_prompt_template(default_prompt_template_file_name: str) -> str:
    # We cannot use importlib.resources with Python 3.9 because of an unfixed bug:
    # https://bugs.python.org/issue44137
    if sys.version_info < (3, 10):
        from importlib_resources import files

        default_prompt_template = (
            files(LLM_JUDGE_PROMPTS_MODULE)
            .joinpath(default_prompt_template_file_name)
            .read_text()
        )
    else:
        import importlib.resources

        default_prompt_template = importlib.resources.read_text(
            LLM_JUDGE_PROMPTS_MODULE,
            default_prompt_template_file_name,
        )

    return default_prompt_template


def _get_prompt_inputs(
    assertion_type: str,
    matching_event: BotUttered,
    ground_truth: Optional[str] = None,
) -> Dict[str, Any]:
    if assertion_type == AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value:
        return {"num_variations": "3", "bot_message": matching_event.text}
    elif assertion_type == AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value:
        ground_truth_event_metadata = matching_event.metadata.get(
            SEARCH_RESULTS_METADATA_KEY, ""
        ) or matching_event.metadata.get(DOMAIN_GROUND_TRUTH_METADATA_KEY, "")

        if isinstance(ground_truth_event_metadata, list):
            ground_truth_event_metadata = "\n".join(ground_truth_event_metadata)

        ground_truth = (
            ground_truth if ground_truth is not None else ground_truth_event_metadata
        )

        return {
            "bot_message": matching_event.text,
            "ground_truth": ground_truth,
        }
    else:
        raise ValueError(f"Invalid assertion type '{assertion_type}'")


def calculate_score(
    assertion_type: str, processed_output: List[Any], score_inputs: ScoreInputs
) -> Tuple[float, str]:
    """Calculate and return the score and justification."""
    if assertion_type == AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value:
        return calculate_relevance_score(processed_output, score_inputs)
    elif assertion_type == AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value:
        return calculate_groundedness_score(processed_output, score_inputs)
    else:
        raise ValueError(f"Invalid assertion type '{assertion_type}'")
