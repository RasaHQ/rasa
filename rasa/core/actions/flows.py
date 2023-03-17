from typing import Text, List, Optional, Union, Any, Dict
import logging
from copy import deepcopy
from pypred import Predicate
from rasa.core.actions import action
from rasa.core.actions.forms import FormAction
from rasa.core.channels import OutputChannel

from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    REQUESTED_SLOT,
    NEXT_STEP,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.slots import TextSlot
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
        self, flow_name: Text, action_endpoint: Optional[EndpointConfig]
    ) -> None:
        """Creates a `FlowAction`.

        Args:
            flow_name: Name of the flow.
            action_endpoint: Endpoint to execute custom actions.
        """
        self._flow_name = flow_name
        super().__init__(flow_name, action_endpoint)

    def name(self) -> Text:
        """Return the flow name."""
        return self._flow_name

    @staticmethod
    def _add_embedded_flows_steps(steps: List[Dict], domain: Domain) -> List[Dict]:
        """Embed flows in the next step."""
        additional_steps = []
        link_steps = list((step for step in steps if step.get("link")))
        for link_step in link_steps:
            embeddedable_flow = domain.flows.get(link_step.get("link", ""), None)

            if not embeddedable_flow:
                raise Exception(f"Flow '{link_step.get('link')}' not found in domain.")

            prefix = link_step.get("link", "") + "_"
            steps_to_embed = deepcopy(embeddedable_flow.get("steps", []))
            if not steps_to_embed:
                continue

            for index, step in enumerate(steps_to_embed):
                step["id"] = link_step.get("id") if index == 0 else prefix + step["id"]
                if step.get("next"):
                    if isinstance(step["next"], str):
                        step["next"] = prefix + step["next"]
                    else:
                        for condition in step["next"]:
                            if condition.get("if"):
                                condition["then"] = prefix + condition["then"]
                else:
                    # Link back to the calling flow
                    if link_step.get("next"):
                        step["next"] = link_step.get("next")
                additional_steps.append(step)
        steps = [step for step in steps if not step.get("link")]
        return steps + additional_steps

    def steps(self, domain: Domain) -> List[Dict]:
        """Return all the steps of the flow."""
        flow = domain.flows.get(self._flow_name)
        if flow:
            return self._add_embedded_flows_steps(flow.get("steps", []), domain)
        return []

    def required_slots(self, domain: Domain) -> List[Text]:
        """A list of required slots that the flow has to fill.

        Returns:
            A list of slot names.
        """
        return list(
            (
                step.get("question", "")
                for step in self.steps(domain)
                if step.get("question")
            )
        )

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
        """Executes form loop after activation."""
        events: List[Event] = []
        """
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
            events = await self._validate_if_required(
                tracker, domain, output_channel, nlg
            )

        if tracker.get_slot("requested_slot") is not None:
            events += [SlotSet(REQUESTED_SLOT, None)]
        if not self._user_rejected_manually(events):
            events += await self._exec_next_step(
                tracker, domain, output_channel, nlg, events_so_far
            )

        return events

    def _eval_predicate(
        self, predicate: Text, domain: Domain, tracker: "DialogueStateTracker"
    ) -> bool:
        """Evaluate a predicate condition."""

        def get_value(
            initial_value: Union[Text, None]
        ) -> Union[Text, float, bool, None]:
            if initial_value and not isinstance(initial_value, str):
                raise ValueError("Slot is not a text slot")
            elif not initial_value:
                return None
            elif initial_value.lower() in ["true", "false"]:
                return initial_value.lower() == "true"
            return float(initial_value) if initial_value.isnumeric() else initial_value

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

    def _get_else_predicate(
        self, next_items: Dict, domain: Domain, tracker: "DialogueStateTracker"
    ) -> Text:
        negated_if_predicates = list(
            (
                f"not ({item.get('if')})"
                for item in next_items
                if item.get("if") is not None
            )
        )
        else_predicate = "and".join(negated_if_predicates)
        return else_predicate

    def _get_next_step(
        self,
        tracker: "DialogueStateTracker",
        domain: Domain,
    ) -> Optional[Dict]:
        """Get the next step to execute."""
        next_step_slot = tracker.get_slot(NEXT_STEP)
        all_steps = self.steps(domain)
        if next_step_slot is None:
            return all_steps[0]
        # If the question is not answered, we return the current step
        elif (
            self._get_step_type(next_step_slot) == "question"
            and tracker.get_slot(next_step_slot.get("question")) is None
        ):
            return next_step_slot
        elif next_step_slot.get("next") is not None:
            next_info = next_step_slot.get("next")
            next_id = None
            if isinstance(next_info, str):
                next_id = next_step_slot.get("next")

            else:
                if_statements = list(
                    (item.get("if") for item in next_info if item.get("if") is not None)
                )
                evaluations = list(
                    (
                        self._eval_predicate(if_statement, domain, tracker)
                        for if_statement in if_statements
                    )
                )
                first_true_index = next(
                    (
                        index
                        for index, evaluation in enumerate(evaluations)
                        if evaluation
                    ),
                    None,
                )
                if first_true_index is not None:
                    next_id = next_info[first_true_index].get("then")
                else:
                    else_case = next(
                        (item for item in next_info if item.get("else")), None
                    )
                    if else_case:
                        next_id = else_case.get("else")

            if next_id is None:
                raise Exception(
                    f"Conditions in step {next_step_slot['id']} "
                    f"are not covering all possibilities"
                )
            # next_steps = list(
            #     (step["id"] for step in all_steps if step["id"] == next_id)
            # )
            next_steps = list(filter(lambda step: step["id"] == next_id, all_steps))
            if len(next_steps) == 1:
                return next_steps[0]
            elif len(next_steps) == 0:
                raise Exception(f"Next step with id {next_id} not found")
            else:
                raise Exception(f"Multiple next steps with id {next_id} found")
        # If the next step is not specified, we assume that the flow is done
        else:
            return None

    @staticmethod
    def _get_step_type(step: Dict) -> Text:
        """Get the type of the step."""
        if step.get("action"):
            return "action"
        elif step.get("question"):
            return "question"
        else:
            raise Exception(f"Step type not specified for step {step}")

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
        events: List[Event] = [SlotSet(NEXT_STEP, next_step)]
        if next_step:
            if FlowAction._get_step_type(next_step) == "question":
                temp_tracker = self._temporary_tracker(tracker, events_so_far, domain)
                slot = next(
                    (
                        slot
                        for slot in domain.slots
                        if slot.name == next_step["question"]
                    )
                )
                events += [SlotSet(REQUESTED_SLOT, slot.name)]
                events += await self._ask_for_slot(
                    domain, nlg, output_channel, slot.name, temp_tracker
                )
                events += [FollowupAction(ACTION_LISTEN_NAME)]
            elif FlowAction._get_step_type(next_step) == "action":
                action_name = next_step.get("action")
                if action_name is None:
                    raise Exception(f"Action not specified for step {next_step}")

                action_to_run = action.action_for_name_or_text(
                    action_name, domain, self.action_endpoint
                )
                events += await action_to_run.run(output_channel, nlg, tracker, domain)
                events += [FollowupAction(self.name())]

        return events

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
