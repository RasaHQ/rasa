from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
    Text,
    Tuple,
    Type,
)

import pandas as pd
import structlog

import rasa.shared.utils.common
from rasa.core.constants import (
    DOMAIN_GROUND_TRUTH_METADATA_KEY,
    UTTER_SOURCE_METADATA_KEY,
)
from rasa.core.policies.enterprise_search_policy import (
    SEARCH_QUERY_METADATA_KEY,
    SEARCH_RESULTS_METADATA_KEY,
)
from rasa.dialogue_understanding.patterns.clarify import FLOW_PATTERN_CLARIFICATION
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
from rasa.shared.exceptions import RasaException
from rasa.utils.common import update_mlflow_log_level
from rasa.utils.json_utils import SetEncoder

if TYPE_CHECKING:
    from rasa.e2e_test.e2e_config import LLMJudgeConfig


structlogger = structlog.get_logger()

DEFAULT_THRESHOLD = 0.5
ELIGIBLE_UTTER_SOURCE_METADATA = [
    "EnterpriseSearchPolicy",
    "ContextualResponseRephraser",
    "IntentlessPolicy",
]


class AssertionType(Enum):
    FLOW_STARTED = "flow_started"
    FLOW_COMPLETED = "flow_completed"
    FLOW_CANCELLED = "flow_cancelled"
    PATTERN_CLARIFICATION_CONTAINS = "pattern_clarification_contains"
    ACTION_EXECUTED = "action_executed"
    SLOT_WAS_SET = "slot_was_set"
    SLOT_WAS_NOT_SET = "slot_was_not_set"
    BOT_UTTERED = "bot_uttered"
    GENERATIVE_RESPONSE_IS_RELEVANT = "generative_response_is_relevant"
    GENERATIVE_RESPONSE_IS_GROUNDED = "generative_response_is_grounded"


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

    flow_id: str
    line: Optional[int] = None

    @classmethod
    def type(cls) -> str:
        return AssertionType.FLOW_STARTED.value

    @staticmethod
    def from_dict(assertion_dict: Dict[Text, Any]) -> FlowStartedAssertion:
        return FlowStartedAssertion(
            flow_id=assertion_dict.get(AssertionType.FLOW_STARTED.value),
            line=assertion_dict.lc.line + 1 if hasattr(assertion_dict, "lc") else None,
        )

    def run(
        self,
        turn_events: List[Event],
        prior_events: List[Event],
        assertion_order_error_message: str = "",
        **kwargs: Any,
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run the flow started assertion on the given events for that user turn."""
        try:
            matching_event = next(
                event
                for event in turn_events
                if isinstance(event, FlowStarted) and event.flow_id == self.flow_id
            )
        except StopIteration:
            error_message = f"Flow with id '{self.flow_id}' did not start."
            error_message += assertion_order_error_message

            return self._generate_assertion_failure(
                error_message, prior_events, turn_events, self.line
            )

        return None, matching_event

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
            error_message += assertion_order_error_message

            return self._generate_assertion_failure(
                error_message, prior_events, turn_events, self.line
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
            error_message += assertion_order_error_message
            return self._generate_assertion_failure(
                error_message, prior_events, turn_events, self.line
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
            error_message += assertion_order_error_message

            return self._generate_assertion_failure(
                error_message, prior_events, turn_events, self.line
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
            error_message += assertion_order_error_message

            return self._generate_assertion_failure(
                error_message, prior_events, turn_events, self.line
            )

        return None, matching_event

    def __hash__(self) -> int:
        return hash(json.dumps(self.as_dict()))


@dataclass
class PatternClarificationContainsAssertion(Assertion):
    """Class for storing the pattern clarification contains assertion."""

    flow_names: Set[str]
    line: Optional[int] = None

    @classmethod
    def type(cls) -> str:
        return AssertionType.PATTERN_CLARIFICATION_CONTAINS.value

    @staticmethod
    def from_dict(
        assertion_dict: Dict[Text, Any],
    ) -> PatternClarificationContainsAssertion:
        return PatternClarificationContainsAssertion(
            flow_names=set(
                assertion_dict.get(
                    AssertionType.PATTERN_CLARIFICATION_CONTAINS.value, []
                )
            ),
            line=assertion_dict.lc.line + 1 if hasattr(assertion_dict, "lc") else None,
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
            error_message += assertion_order_error_message

            return self._generate_assertion_failure(
                error_message, prior_events, turn_events, self.line
            )

        actual_flow_names = set(matching_event.metadata.get("names", set()))
        if actual_flow_names != self.flow_names:
            error_message = (
                f"'{FLOW_PATTERN_CLARIFICATION}' pattern did not contain "
                f"the expected options. Expected options: {self.flow_names}. "
            )
            error_message += assertion_order_error_message

            return self._generate_assertion_failure(
                error_message, prior_events, turn_events, self.line
            )

        return None, matching_event

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
        try:
            matching_event = next(
                event
                for event in turn_events
                if isinstance(event, ActionExecuted)
                and event.action_name == self.action_name
            )
        except StopIteration:
            error_message = f"Action '{self.action_name}' did not execute."
            error_message += assertion_order_error_message

            return self._generate_assertion_failure(
                error_message, prior_events, turn_events, self.line
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

        for slot in self.slots:
            matching_events = [
                event
                for event in turn_events
                if isinstance(event, SlotSet) and event.key == slot.name
            ]
            if not matching_events:
                error_message = f"Slot '{slot.name}' was not set."
                error_message += assertion_order_error_message

                return self._generate_assertion_failure(
                    error_message, prior_events, turn_events, slot.line
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
                error_message += assertion_order_error_message

                return self._generate_assertion_failure(
                    error_message, prior_events, turn_events, slot.line
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
                error_message += assertion_order_error_message

                return self._generate_assertion_failure(
                    error_message, prior_events, turn_events, slot.line
                )

            if matching_event.value == slot.value:
                error_message = (
                    f"Slot '{slot.name}' was set to '{slot.value}' "
                    f"but it should not have been set."
                )
                error_message += assertion_order_error_message

                return self._generate_assertion_failure(
                    error_message, prior_events, turn_events, slot.line
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

        if self.utter_name is not None:
            try:
                matching_event = next(
                    event
                    for event in turn_events
                    if isinstance(event, BotUttered)
                    and event.metadata.get("utter_action") == self.utter_name
                )
            except StopIteration:
                error_message = f"Bot did not utter '{self.utter_name}' response."
                error_message += assertion_order_error_message

                return self._generate_assertion_failure(
                    error_message, prior_events, turn_events, self.line
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
                error_message = (
                    f"Bot did not utter any response which "
                    f"matches the provided text pattern "
                    f"'{self.text_matches}'."
                )
                error_message += assertion_order_error_message

                return self._generate_assertion_failure(
                    error_message, prior_events, turn_events, self.line
                )

        if self.buttons:
            try:
                matching_event = next(
                    event
                    for event in turn_events
                    if isinstance(event, BotUttered) and self._buttons_match(event)
                )
            except StopIteration:
                error_message = (
                    "Bot did not utter any response with the expected buttons."
                )
                error_message += assertion_order_error_message
                return self._generate_assertion_failure(
                    error_message, prior_events, turn_events, self.line
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
class GenerativeResponseMixin(Assertion):
    """Mixin class for storing generative response assertions."""

    threshold: float = DEFAULT_THRESHOLD
    utter_name: Optional[str] = None
    line: Optional[int] = None
    metric_adjective: Optional[str] = None
    metric_name: Optional[str] = None
    mlflow_metric: Callable = print

    @classmethod
    def type(cls) -> str:
        return ""

    def _get_ground_truth(self, matching_event: BotUttered) -> str:
        raise NotImplementedError

    def as_dict(self) -> Dict[str, Any]:
        data = super().as_dict()
        data.pop("metric_name")
        data.pop("metric_adjective")
        data.pop("mlflow_metric")

        return data

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
        import mlflow

        # we need to configure the log level for mlflow
        # after a local import to avoid unnecessary logs
        update_mlflow_log_level()

        # extract user question from event if available
        user_question_from_event = matching_event.metadata.get(
            SEARCH_QUERY_METADATA_KEY
        )
        user_question = (
            user_question_from_event if user_question_from_event else step_text
        )

        ground_truth = self._get_ground_truth(matching_event)

        eval_data = pd.DataFrame(
            {
                "inputs": [user_question],
                "ground_truth": [ground_truth],
                "predictions": [matching_event.text],
            }
        )

        model_uri = llm_judge_config.get_model_uri()

        structlogger.debug(
            f"generative_response_is_{self.metric_adjective}_assertion.run_llm_evaluation",
            model_uri=model_uri,
        )

        with mlflow.start_run():
            results = mlflow.evaluate(
                data=eval_data,
                targets="ground_truth",
                predictions="predictions",
                model_type="question-answering",
                evaluators="default",
                extra_metrics=[
                    self.mlflow_metric(model_uri),
                ],
            )

        # Evaluation result for each data record is available in `results.tables`.
        eval_table = results.tables["eval_results_table"]
        score = eval_table.iloc[0][f"{self.metric_name}/v1/score"]
        justification = eval_table.iloc[0][f"{self.metric_name}/v1/justification"]

        # convert 1-5 score to 0-1 float
        score = score * 20 / 100 if score is not None else 0

        structlogger.debug(
            f"generative_response_is_{self.metric_adjective}_assertion.run_results",
            matching_event=repr(matching_event),
            score=score,
            justification=justification,
        )

        if score < self.threshold:
            error_message = (
                f"Generative response '{matching_event.text}' "
                f"given to the user input '{user_question}' "
                f"was not {self.metric_adjective}. "
                f"Expected score to be above '{self.threshold}' threshold, "
                f"but was '{score}'. The explanation for this score is: "
                f"{justification}."
            )
            error_message += assertion_order_error_message

            return self._generate_assertion_failure(
                error_message, prior_events, turn_events, self.line
            )

        return None, matching_event

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
            error_message += assertion_order_error_message

            return self._generate_assertion_failure(
                error_message, prior_events, turn_events, self.line
            )

        return self._run_llm_evaluation(
            matching_event,
            step_text,
            llm_judge_config,
            assertion_order_error_message,
            prior_events,
            turn_events,
        )

    def _run_assertion_for_multiple_generative_responses(
        self,
        matching_events: List[BotUttered],
        step_text: str,
        llm_judge_config: "LLMJudgeConfig",
        assertion_order_error_message: str,
        prior_events: List[Event],
        turn_events: List[Event],
    ) -> Tuple[Optional[AssertionFailure], Optional[Event]]:
        """Run LLM evaluation for multiple bot utterances."""
        structlogger.debug(
            f"generative_response_is_{self.metric_adjective}_assertion.run",
            event_info="Multiple generative responses found, "
            "we will evaluate each of the responses.",
        )

        passing_events = set()
        for event in matching_events:
            failure, event_result = self._run_llm_evaluation(
                event,
                step_text,
                llm_judge_config,
                assertion_order_error_message,
                prior_events,
                turn_events,
            )
            if event_result is not None:
                passing_events.add(event_result)
        else:
            if not passing_events:
                error_message = (
                    f"None of the generative responses issued by either the "
                    f"Enterprise Search Policy, IntentlessPolicy or the "
                    f"Contextual Response Rephraser were {self.metric_adjective}."
                )
                error_message += assertion_order_error_message

                return self._generate_assertion_failure(
                    error_message, prior_events, turn_events, self.line
                )

        return None, list(passing_events)[-1]

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
            turn_events
        )

        if not matching_events:
            error_message = (
                "No generative response issued by either the Enterprise Search Policy, "
                "IntentlessPolicy or the Contextual Response Rephraser was found, "
                "but one was expected."
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
            return self._run_assertion_for_multiple_generative_responses(
                matching_events,
                step_text,
                llm_judge_config,
                assertion_order_error_message,
                prior_events,
                turn_events,
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

    def _get_ground_truth(self, matching_event: BotUttered) -> str:
        return ""

    @classmethod
    def type(cls) -> str:
        return AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value

    @staticmethod
    def from_dict(
        assertion_dict: Dict[Text, Any],
    ) -> GenerativeResponseIsRelevantAssertion:
        import mlflow

        assertion_dict = assertion_dict.get(
            AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value, {}
        )
        return GenerativeResponseIsRelevantAssertion(
            threshold=assertion_dict.get("threshold", DEFAULT_THRESHOLD),
            utter_name=assertion_dict.get("utter_name"),
            line=assertion_dict.lc.line + 1 if hasattr(assertion_dict, "lc") else None,
            metric_name="answer_relevance",
            metric_adjective="relevant",
            mlflow_metric=mlflow.metrics.genai.answer_relevance,
        )

    def __hash__(self) -> int:
        return hash(json.dumps(self.as_dict()))


@dataclass
class GenerativeResponseIsGroundedAssertion(GenerativeResponseMixin):
    """Class for storing the generative response is grounded assertion."""

    ground_truth: Optional[str] = None

    @classmethod
    def type(cls) -> str:
        return AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value

    @staticmethod
    def from_dict(
        assertion_dict: Dict[Text, Any],
    ) -> GenerativeResponseIsGroundedAssertion:
        import mlflow

        assertion_dict = assertion_dict.get(
            AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value, {}
        )
        return GenerativeResponseIsGroundedAssertion(
            threshold=assertion_dict.get("threshold", DEFAULT_THRESHOLD),
            utter_name=assertion_dict.get("utter_name"),
            ground_truth=assertion_dict.get("ground_truth"),
            line=assertion_dict.lc.line + 1 if hasattr(assertion_dict, "lc") else None,
            metric_name="answer_correctness",
            metric_adjective="grounded",
            mlflow_metric=mlflow.metrics.genai.answer_correctness,
        )

    def __hash__(self) -> int:
        return hash(json.dumps(self.as_dict()))

    def _get_ground_truth(self, matching_event: BotUttered) -> str:
        # extract ground truth from event if available or use the provided ground truth
        ground_truth_event_metadata = matching_event.metadata.get(
            SEARCH_RESULTS_METADATA_KEY, ""
        ) or matching_event.metadata.get(DOMAIN_GROUND_TRUTH_METADATA_KEY, "")

        if isinstance(ground_truth_event_metadata, list):
            ground_truth_event_metadata = "\n".join(ground_truth_event_metadata)

        ground_truth = (
            self.ground_truth
            if self.ground_truth is not None
            else ground_truth_event_metadata
        )

        return ground_truth


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


def _find_matching_generative_events(turn_events: List[Event]) -> List[BotUttered]:
    """Find the matching events for the generative response assertions."""
    return [
        event
        for event in turn_events
        if isinstance(event, BotUttered)
        and event.metadata.get(UTTER_SOURCE_METADATA_KEY)
        in ELIGIBLE_UTTER_SOURCE_METADATA
    ]
