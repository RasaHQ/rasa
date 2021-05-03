from typing import Text, List, Optional, Union, Any, Dict, Tuple, Set
import logging
import json

from rasa.core.actions import action
from rasa.core.actions.loops import LoopAction
from rasa.core.channels import OutputChannel
from rasa.shared.core.domain import Domain, InvalidDomain, SlotMapping, State

from rasa.core.actions.action import ActionExecutionRejection, RemoteAction
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    REQUESTED_SLOT,
    LOOP_INTERRUPTED,
)
from rasa.shared.constants import UTTER_PREFIX
from rasa.shared.core.events import (
    Event,
    SlotSet,
    ActionExecuted,
    ActiveLoop,
    ActionExecutionRejected,
)
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig

from rasa.shared.nlu.state_machine.state_machine_models import (
    Intent,
    Slot,
    Utterance,
)

from rasa.shared.nlu.state_machine.state_machine_state import (
    Response,
    StateMachineState,
    Transition,
)


import random

logger = logging.getLogger(__name__)


class StateMachineAction(LoopAction):
    """Action which implements and executes the form logic."""

    def __init__(self, action_endpoint: Optional[EndpointConfig]) -> None:
        """Creates a `StateMachineAction`.

        Args:
            form_name: Name of the form.
            action_endpoint: Endpoint to execute custom actions.
        """
        # self._form_name = form_name
        self.action_endpoint = action_endpoint
        # creating it requires domain, which we don't have in init
        # we'll create it on the first call
        self._unique_entity_mappings = None

    def name(self) -> Text:
        return "action_state_machine_action"

    def required_slots(self, domain: Domain) -> List[Text]:
        """A list of required slots that the form has to fill.

        Returns:
            A list of slot names.
        """
        return list(domain.slot_mapping_for_form(self.name()).keys())

    async def validate_slots(
        self,
        slot_candidates: Dict[Text, Any],
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Event]:
        """Validate the extracted slots.

        If a custom action is available for validating the slots, we call it to validate
        them. Otherwise there is no validation.

        Args:
            slot_candidates: Extracted slots which are candidates to fill the slots
                required by the form.
            tracker: The current conversation tracker.
            domain: The current model domain.
            output_channel: The output channel which can be used to send messages
                to the user.
            nlg:  `NaturalLanguageGenerator` to use for response generation.

        Returns:
            The validation events including potential bot messages and `SlotSet` events
            for the validated slots.
        """
        logger.debug(f"Validating extracted slots: {slot_candidates}")
        events = [
            SlotSet(slot_name, value)
            for slot_name, value in slot_candidates.items()
        ]

        validate_name = f"validate_{self.name()}"

        if validate_name not in domain.action_names_or_texts:
            return events

        _tracker = self._temporary_tracker(tracker, events, domain)
        _action = RemoteAction(validate_name, self.action_endpoint)
        validate_events = await _action.run(
            output_channel, nlg, _tracker, domain
        )

        validated_slot_names = [
            event.key
            for event in validate_events
            if isinstance(event, SlotSet)
        ]

        # If the custom action doesn't return a SlotSet event for an extracted slot
        # candidate we assume that it was valid. The custom action has to return a
        # SlotSet(slot_name, None) event to mark a Slot as invalid.
        return validate_events + [
            event for event in events if event.key not in validated_slot_names
        ]

    def _temporary_tracker(
        self,
        current_tracker: DialogueStateTracker,
        additional_events: List[Event],
        domain: Domain,
    ) -> DialogueStateTracker:
        return DialogueStateTracker.from_events(
            current_tracker.sender_id,
            current_tracker.events_after_latest_restart()
            # Insert SlotSet event to make sure REQUESTED_SLOT belongs to active form.
            # + [SlotSet(REQUESTED_SLOT, self.get_slot_to_fill(current_tracker))]
            # Insert form execution event so that it's clearly distinguishable which
            # events were newly added.
            + [ActionExecuted(self.name())] + additional_events,
            slots=domain.slots,
        )

    def _user_rejected_manually(self, validation_events: List[Event]) -> bool:
        """Checks if user rejected the form execution during a slot_validation.

        Args:
            validation_events: Events returned by the custom slot_validation action

        Returns:
            True if the validation_events include an ActionExecutionRejected event,
            else False.
        """
        return any(
            isinstance(event, ActionExecutionRejected)
            for event in validation_events
        )

    @staticmethod
    def _get_valid_slots(
        slots: List[Slot],
        tracker: DialogueStateTracker,
        only_empty_slots: bool = False,
    ) -> List[Slot]:
        valid_slots = slots

        # Get only unfilled slots
        if only_empty_slots:
            valid_slots = [
                slot
                for slot in valid_slots
                if tracker.get_slot(slot.name) == None
            ]

        # Get slots that pass conditions
        valid_slots = [
            slot
            for slot in valid_slots
            if not slot.condition
            or (slot.condition and slot.condition.is_valid(tracker))
        ]

        return valid_slots

    @staticmethod
    def _get_slot_values(
        slots: List[Slot], tracker: "DialogueStateTracker"
    ) -> Dict[str, Any]:
        slot_values = {}

        # Get valid slots from latest utterance
        valid_slots = StateMachineAction._get_valid_slots(
            slots=slots, tracker=tracker, only_empty_slots=False
        )

        for slot in valid_slots:
            last_bot_uttered_action_name = (
                tracker.latest_bot_utterance.metadata.get("utter_action")
            )

            if slot.only_fill_when_prompted:
                if not last_bot_uttered_action_name or (
                    last_bot_uttered_action_name
                    not in [action.name for action in slot.prompt_actions]
                ):
                    continue

            # Extract values using entities
            values_for_slots: List[Any] = [
                StateMachineAction.get_entity_value(
                    entity, tracker, None, None
                )
                for entity in slot.entities
            ]

            # Extract values using intents
            last_intent_name = tracker.latest_message.intent.get("name")
            if last_intent_name:
                for slot_intent, value in slot.intents.items():
                    if isinstance(slot_intent, Intent):
                        if slot_intent.name == last_intent_name:
                            values_for_slots.append(value)
                    elif isinstance(slot_intent, str):
                        if slot_intent == last_intent_name:
                            values_for_slots.append(value)

            # Filter out None's
            values_for_slots = [
                value for value in values_for_slots if value is not None
            ]

            if len(values_for_slots) > 0:
                # Take first entity extracted
                slot_values.update({slot.name: values_for_slots[0]})

        return slot_values

    @staticmethod
    def _get_response_action_names(
        responses: List[Response], tracker: "DialogueStateTracker"
    ) -> List[str]:
        valid_responses = [
            response
            for response in responses
            if response.condition.is_valid(tracker=tracker)
        ]

        # valid_responses_action_names = [
        #     valid_response.actions[
        #         random.randint(0, len(valid_response.actions) - 1)
        #     ].name
        #     for valid_response in valid_responses
        #     if len(valid_response.actions) > 0
        # ]

        valid_responses_action_names = [
            action.name
            for valid_response in valid_responses
            for action in valid_response.actions
        ]

        return valid_responses_action_names

    async def _get_slot_fill_events(
        self,
        slots: List[Slot],
        slot_fill_utterances: List[Utterance],
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Event]:
        slot_values = StateMachineAction._get_slot_values(
            slots=slots, tracker=tracker
        )

        # Convert slot values to events
        validation_events = await self.validate_slots(
            slot_values, tracker, domain, output_channel, nlg
        )

        # Create temporary tracker with the validation events applied
        # Otherwise, the slots will not be set
        temp_tracker = self._temporary_tracker(
            tracker, validation_events, domain
        )

        # Find valid utterances
        # TODO: Handle slots that are filled but not uttered.
        slot_filled_utterance: Optional[Utterance] = None

        if len(slot_values) > 0:
            number_of_slots_in_utterance = 0
            for utterance in slot_fill_utterances:
                uttered_slot_names = [
                    slot_name
                    for slot_name in slot_values.keys()
                    if f"{{{slot_name}}}" in utterance.text
                ]

                if len(uttered_slot_names) > number_of_slots_in_utterance:
                    slot_filled_utterance = utterance
                    number_of_slots_in_utterance = len(uttered_slot_names)

        # Add slot_filled_action
        slot_filled_action_events: List[Event] = []
        if slot_filled_utterance:
            utterance_action = action.action_for_name_or_text(
                slot_filled_utterance.name,
                domain,
                self.action_endpoint,
            )

            slot_filled_action_events += await utterance_action.run(
                output_channel, nlg, temp_tracker, domain
            )

        return validation_events + slot_filled_action_events

    async def _get_responses_events(
        self,
        responses: List[Response],
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Event]:
        valid_responses_action_names = (
            StateMachineAction._get_response_action_names(responses, tracker)
        )

        valid_responses_actions = [
            action.action_for_name_or_text(
                action_name,
                domain,
                self.action_endpoint,
            )
            for action_name in valid_responses_action_names
        ]

        events: List[Event] = []

        for valid_responses_action in valid_responses_actions:
            events += await valid_responses_action.run(
                output_channel, nlg, tracker, domain
            )

        return events

    async def _get_next_slot_events(
        self,
        slots: List[Slot],
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Event]:
        # Get non-filled slots
        empty_slots = StateMachineAction._get_valid_slots(
            slots=slots, tracker=tracker, only_empty_slots=True
        )

        # Get next slot utterance
        next_slot_prompt_action: Optional[str] = None

        if len(empty_slots) > 0:
            empty_slot = empty_slots[0]
            if len(empty_slot.prompt_actions) == 0:
                raise ActionExecutionRejection(
                    self.name(),
                    f"No prompt actions found for slot {empty_slot.name}",
                )

            randomIndex = random.randint(0, len(empty_slot.prompt_actions) - 1)
            next_slot_utterance_name = empty_slot.prompt_actions[
                randomIndex
            ].name
            next_slot_prompt_action = action.action_for_name_or_text(
                next_slot_utterance_name,
                domain,
                self.action_endpoint,
            )

            return await next_slot_prompt_action.run(
                output_channel, nlg, tracker, domain
            )

        return []

    async def _get_transitions_events(
        self,
        transitions: List[Transition],
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Event]:
        return []

    async def validate(
        self,
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Event]:
        """Extract and validate value of requested slot.

        If nothing was extracted reject execution of the form action.
        Subclass this method to add custom validation and rejection logic
        """

        # Get current state info
        state_machine_state: StateMachineState = (
            domain.active_state_machine_state
        )

        if not state_machine_state:
            raise ActionExecutionRejection(
                self.name(),
                f"No actions found for action {self.name()}",
            )

        # Using the last user message, fill slots
        events: List[Event] = []
        events += await self._get_slot_fill_events(
            slots=state_machine_state.slots,
            slot_fill_utterances=state_machine_state.slot_fill_utterances,
            tracker=tracker,
            domain=domain,
            output_channel=output_channel,
            nlg=nlg,
        )

        # Create temporary tracker with the validation events applied
        # Otherwise, the slots will not be set
        tracker_with_slots_set = self._temporary_tracker(
            tracker, events, domain
        )

        # Check if conditions match any Responses
        events += await self._get_responses_events(
            responses=state_machine_state.responses,
            tracker=tracker_with_slots_set,
            domain=domain,
            output_channel=output_channel,
            nlg=nlg,
        )

        # Check if intent matches any Transitions
        events += await self._get_transitions_events(
            transitions=state_machine_state.transitions,
            tracker=tracker_with_slots_set,
            domain=domain,
            output_channel=output_channel,
            nlg=nlg,
        )

        if len(events) > 0:
            # Ask for next slot
            events += await self._get_next_slot_events(
                slots=state_machine_state.slots,
                tracker=tracker_with_slots_set,
                domain=domain,
                output_channel=output_channel,
                nlg=nlg,
            )

            return events
        else:
            raise ActionExecutionRejection(
                self.name(),
                f"No actions found for action {self.name()}",
            )

    async def _validate_if_required(
        self,
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Event]:
        """Return a list of events from `self.validate(...)`.

        Validation is required if:
           - the form is active
           - the form is called after `action_listen`
           - form validation was not cancelled
        """
        # no active_loop means that it is called during activation
        needs_validation = not tracker.active_loop or (
            tracker.latest_action_name == ACTION_LISTEN_NAME
            and not tracker.active_loop.get(LOOP_INTERRUPTED, False)
        )

        if needs_validation:
            logger.debug(f"Validating user input '{tracker.latest_message}'.")
            return await self.validate(tracker, domain, output_channel, nlg)
        else:
            # Needed to determine which slots to request although there are no slots
            # to actually validate, which happens when coming back to the form after
            # an unhappy path
            return await self.validate_slots(
                {}, tracker, domain, output_channel, nlg
            )

    @staticmethod
    def _should_request_slot(
        tracker: "DialogueStateTracker", slot_name: Text
    ) -> bool:
        """Check whether form action should request given slot"""

        return tracker.get_slot(slot_name) is None

    async def activate(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        """Activate form if the form is called for the first time.

        If activating, validate any required slots that were filled before
        form activation and return `Form` event with the name of the form, as well
        as any `SlotSet` events from validation of pre-filled slots.

        Args:
            output_channel: The output channel which can be used to send messages
                to the user.
            nlg: `NaturalLanguageGenerator` to use for response generation.
            tracker: Current conversation tracker of the user.
            domain: Current model domain.

        Returns:
            Events from the activation.
        """

        logger.debug(f"Activated the form '{self.name()}'.")
        # collect values of required slots filled before activation
        prefilled_slots = {}

        for slot_name in self.required_slots(domain):
            if not self._should_request_slot(tracker, slot_name):
                prefilled_slots[slot_name] = tracker.get_slot(slot_name)

        if not prefilled_slots:
            logger.debug("No pre-filled required slots to validate.")
            return []

        logger.debug(
            f"Validating pre-filled required slots: {prefilled_slots}"
        )
        return await self.validate_slots(
            prefilled_slots, tracker, domain, output_channel, nlg
        )

    async def do(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        events_so_far: List[Event],
    ) -> List[Event]:
        events = await self._validate_if_required(
            tracker, domain, output_channel, nlg
        )

        return events

    async def is_done(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        events_so_far: List[Event],
    ) -> bool:
        if any(
            isinstance(event, ActionExecutionRejected)
            for event in events_so_far
        ):
            return False

        # Custom validation actions can decide to terminate the loop early by
        # setting the requested slot to `None` or setting `ActiveLoop(None)`.
        # We explicitly check only the last occurrences for each possible termination
        # event instead of doing `return event in events_so_far` to make it possible
        # to override termination events which were returned earlier.
        return (
            next(
                (
                    event
                    for event in reversed(events_so_far)
                    if isinstance(event, SlotSet)
                    and event.key == REQUESTED_SLOT
                ),
                None,
            )
            == SlotSet(REQUESTED_SLOT, None)
            or next(
                (
                    event
                    for event in reversed(events_so_far)
                    if isinstance(event, ActiveLoop)
                ),
                None,
            )
            == ActiveLoop(None)
        )

    async def deactivate(self, *args: Any, **kwargs: Any) -> List[Event]:
        logger.debug(f"Deactivating the form '{self.name()}'")
        return []

    @staticmethod
    def get_entity_value(
        name: Text,
        tracker: "DialogueStateTracker",
        role: Optional[Text] = None,
        group: Optional[Text] = None,
    ) -> Any:
        """Extract entities for given name and optional role and group.

        Args:
            name: entity type (name) of interest
            tracker: the tracker
            role: optional entity role of interest
            group: optional entity group of interest

        Returns:
            Value of entity.
        """

        # Return None if latest entity values were gathered before the active loop was set
        if not tracker.active_loop_name:
            return None

        # list is used to cover the case of list slot type
        value = list(
            tracker.get_latest_entity_values(
                name, entity_group=group, entity_role=role
            )
        )
        if len(value) == 0:
            value = None
        elif len(value) == 1:
            value = value[0]
        return value
