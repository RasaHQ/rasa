from typing import Text, List, Optional, Union, Any
import logging
from pypred import Predicate
from rasa.core.actions import action
from rasa.core.actions.forms import FormAction
from rasa.core.channels import OutputChannel
from rasa.shared.constants import FLOW_PREFIX
from rasa.shared.core.flows.flow import (
    ActionFlowStep,
    ElseFlowLink,
    Flow,
    FlowLinks,
    FlowStep,
    FlowsList,
    IfFlowLink,
    QuestionFlowStep,
    StaticFlowLink,
)

from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    REQUESTED_SLOT,
    NEXT_STEP,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.slots import Slot, TextSlot
from rasa.shared.core.events import (
    Event,
    SlotSet,
    FollowupAction,
    ActiveLoop,
    ActionExecutionRejected,
)
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


class FlowAction(FormAction):
    """Action which implements and executes the form logic."""

    def __init__(
        self, flow_action_name: Text, action_endpoint: Optional[EndpointConfig]
    ) -> None:
        """Creates a `FlowAction`.

        Args:
            flow_action_name: Name of the flow.
            action_endpoint: Endpoint to execute custom actions.
        """
        self._flow_name = flow_action_name[len(FLOW_PREFIX) :]
        self._flow_action_name = flow_action_name
        super().__init__(self._flow_action_name, action_endpoint)

    def name(self) -> Text:
        """Return the flow name."""
        return self._flow_action_name

    @staticmethod
    def flow_by_name(domain: Domain, name: Optional[Text]) -> Optional[Flow]:
        """Return the flow with the given name."""
        # TODO: this is pretty hacky. we attach it to the domain because
        #    we have access to the domain in the action.
        if not name:
            return None

        if not hasattr(domain, "flows"):
            return None

        flows: FlowsList = domain.flows
        for flow in flows.underlying_flows:
            if flow.flow == name:
                return flow
        else:
            return None

    def steps(self, domain: Domain) -> List[FlowStep]:
        """Return all the steps of the flow."""
        flow = self.flow_by_name(domain, self._flow_name)
        if flow:
            return flow.steps
        return []

    def first_step(self, domain: Domain) -> Optional[FlowStep]:
        """Return the first step of the flow."""
        steps = self.steps(domain)
        return steps[0] if steps else None

    def step_by_id(self, domain: Domain, step_id: Text) -> FlowStep:
        """Return the step with the given id."""
        for step in self.steps(domain):
            if step.id == step_id:
                return step

        else:
            raise ValueError(f"Step with id '{step_id}' not found.")

    def required_slots(self, domain: Domain) -> List[Text]:
        """A list of required slots that the flow has to fill.

        Returns:
            A list of slot names.
        """
        return [
            step.question
            for step in self.flow_by_name(domain, self._flow_name).steps
            if step.question
        ]

    async def activate(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        """Activate flow if the flow is called for the first time.

        If activating, run action_extract_slots to fill slots with
        mapping conditions from trigger intents.
        Validate any required slots that can be filled, and return any `SlotSet`
        events from the extraction and validation of these pre-filled slots.

        Args:
            output_channel: The output channel which can be used to send messages
                to the user.
            nlg: `NaturalLanguageGenerator` to use for response generation.
            tracker: Current conversation tracker of the user.
            domain: Current model domain.

        Returns:
            Events from the activation.
        """
        logger.debug(f"Activated the flow '{self.name()}'.")
        return []

    async def do(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        events_so_far: List[Event],
    ) -> List[Event]:
        """Executes form loop after activation.

        Call to validation is not required when the slots are already validated
        at the time of form activation.
        events_so_far:
            - empty when slots have not been validated.
            - has SlotSet objects when already validated.
            - ActiveLoop object when events have not been validated.
        Hence the events are filtered to remove ActiveLoop object that was added
        at the time of form activation.
        """
        filtered_events = [
            event for event in events_so_far if not isinstance(event, ActiveLoop)
        ]
        if not filtered_events:
            new_events = await self._validate_if_required(
                tracker, domain, output_channel, nlg
            )
        else:
            new_events = []

        if tracker.get_slot("requested_slot") is not None:
            new_events.append(SlotSet(REQUESTED_SLOT, None))

        if not self._user_rejected_manually(new_events):
            step_events = await self._exec_next_step(
                tracker, domain, output_channel, nlg, events_so_far
            )
            new_events.extend(step_events)

        return new_events

    def _is_condition_satisfied(
        self, predicate: Text, domain: Domain, tracker: "DialogueStateTracker"
    ) -> bool:
        """Evaluate a predicate condition."""

        def get_value(
            initial_value: Union[Text, None]
        ) -> Union[Text, float, bool, None]:
            if initial_value and not isinstance(initial_value, str):
                raise ValueError("Slot is not a text slot")

            if not initial_value:
                return None

            if initial_value.lower() in ["true", "false"]:
                return initial_value.lower() == "true"

            if initial_value.isnumeric():
                return float(initial_value)

            return initial_value

        text_slots = dict(
            {
                slot.name: get_value(tracker.get_slot(slot.name))
                for slot in domain.slots
                if isinstance(slot, TextSlot)
            }
        )
        p = Predicate(predicate)
        evaluation, _ = p.analyze(text_slots)
        return evaluation

    def _evaluate_flow_links(
        self, next: FlowLinks, domain: Domain, tracker: "DialogueStateTracker"
    ) -> Optional[FlowLinks]:
        """Evaluate the flow links of a step."""
        if len(next.links) == 1 and isinstance(next.links[0], StaticFlowLink):
            return next.links[0].target

        # evaluate if conditions
        for link in next.links:
            if isinstance(link, IfFlowLink):
                if self._is_condition_satisfied(link.condition, domain, tracker):
                    return link.target

        # evaluate else condition
        for link in next.links:
            if isinstance(link, ElseFlowLink):
                return link.target

        return None

    def _get_next_step(
        self,
        tracker: "DialogueStateTracker",
        domain: Domain,
    ) -> Optional[FlowStep]:
        """Get the next step to execute."""
        if not (current_step_id := tracker.get_slot(NEXT_STEP)):
            # If the next step is not set, we return the first step
            # if there is one
            return self.first_step(domain)

        current_step = self.step_by_id(domain, current_step_id)

        # If the question is not answered, we return the current step
        if (
            isinstance(current_step, QuestionFlowStep)
            and tracker.get_slot(current_step.question) is None
        ):
            return current_step

        # If the next step is not specified, we assume that the flow is done
        if not current_step.has_next():
            return None

        next_id = self._evaluate_flow_links(current_step.next, domain, tracker)
        if next_id is None:
            raise Exception(
                f"Conditions in step {current_step.id} "
                f"are not covering all possibilities"
            )
        return self.step_by_id(domain, next_id)

    def _slot_for_question(self, question: Text, domain: Domain) -> Slot:
        """Find the slot for a question."""
        for slot in domain.slots:
            if slot.name == question:
                return slot
        else:
            raise Exception(f"Question '{question}' does not map to an existing slot.")

    async def _exec_next_step(
        self,
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        events_so_far: List[Event],
    ) -> List[Union[SlotSet, Event]]:
        """Request the next slot and response if needed, else return `None`."""
        next_step = self._get_next_step(tracker, domain)
        if not next_step:
            return [SlotSet(NEXT_STEP, None)]

        if isinstance(next_step, QuestionFlowStep):
            temp_tracker = self._temporary_tracker(tracker, events_so_far, domain)
            slot = self._slot_for_question(next_step.question, domain)

            slot_ask_events = await self._ask_for_slot(
                domain, nlg, output_channel, slot.name, temp_tracker
            )
            return [
                SlotSet(NEXT_STEP, next_step.id),
                SlotSet(REQUESTED_SLOT, slot.name),
                *slot_ask_events,
                FollowupAction(ACTION_LISTEN_NAME),
            ]
        elif isinstance(next_step, ActionFlowStep):

            if not (action_name := next_step.action):
                raise Exception(f"Action not specified for step {next_step}")

            action_to_run = action.action_for_name_or_text(
                action_name, domain, self.action_endpoint
            )
            # TODO: this is black magic, we should find a better way to do this
            # an action shouldn't run other actions.

            action_events = await action_to_run.run(
                output_channel, nlg, tracker, domain
            )
            return [
                SlotSet(NEXT_STEP, next_step.id),
                *action_events,
                FollowupAction(self.name()),
            ]
        else:
            return [SlotSet(NEXT_STEP, next_step.id)]

    async def is_done(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        events_so_far: List[Event],
    ) -> bool:
        """Checks if loop can be terminated."""
        if any(isinstance(event, ActionExecutionRejected) for event in events_so_far):
            return False

        # Custom validation actions can decide to terminate the loop early by
        # setting the requested slot to `None` or setting `ActiveLoop(None)`.
        # We explicitly check only the last occurrences for each possible termination
        # event instead of doing `return event in events_so_far` to make it possible
        # to override termination events which were returned earlier.
        there_is_no_next_step = next(
            (
                event
                for event in reversed(events_so_far)
                if isinstance(event, SlotSet) and event.key == NEXT_STEP
            ),
            None,
        ) == SlotSet(NEXT_STEP, None)

        there_is_no_active_loop = next(
            (
                event
                for event in reversed(events_so_far)
                if isinstance(event, ActiveLoop)
            ),
            None,
        ) == ActiveLoop(None)

        return there_is_no_next_step or there_is_no_active_loop

    async def deactivate(self, *args: Any, **kwargs: Any) -> List[Event]:
        """Deactivates form."""
        logger.debug(f"Deactivating the flow '{self.name()}'")
        return []
