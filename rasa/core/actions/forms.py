import copy
import itertools
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Text, Union

import structlog

from rasa.core.actions import action
from rasa.core.actions.action import RemoteAction
from rasa.core.actions.action_exceptions import ActionExecutionRejection
from rasa.core.actions.loops import LoopAction
from rasa.core.channels import OutputChannel
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.shared.constants import UTTER_PREFIX
from rasa.shared.core.constants import (
    ACTION_EXTRACT_SLOTS,
    ACTION_LISTEN_NAME,
    KEY_MAPPING_TYPE,
    REQUESTED_SLOT,
    SLOT_MAPPINGS,
    SlotMappingType,
)
from rasa.shared.core.domain import KEY_SLOTS, Domain
from rasa.shared.core.events import (
    ActionExecuted,
    ActionExecutionRejected,
    ActiveLoop,
    Event,
    Restarted,
    SlotSet,
)
from rasa.shared.core.slot_mappings import SlotMapping
from rasa.shared.core.slots import ListSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from rasa.shared.core.slot_mappings import SlotMapping

logger = logging.getLogger(__name__)
structlogger = structlog.get_logger()


class FormAction(LoopAction):
    """Action which implements and executes the form logic."""

    def __init__(
        self, form_name: Text, action_endpoint: Optional[EndpointConfig]
    ) -> None:
        """Creates a `FormAction`.

        Args:
            form_name: Name of the form.
            action_endpoint: Endpoint to execute custom actions.
        """
        self._form_name = form_name
        self.action_endpoint = action_endpoint
        # creating it requires domain, which we don't have in init
        # we'll create it on the first call
        self._unique_entity_mappings: Set[Text] = set()
        self._have_unique_entity_mappings_been_initialized = False

    def name(self) -> Text:
        """Return the form name."""
        return self._form_name

    def required_slots(self, domain: Domain) -> List[Text]:
        """A list of required slots that the form has to fill.

        Returns:
            A list of slot names.
        """
        return domain.required_slots_for_form(self.name())

    def from_entity(
        self,
        entity: Text,
        intent: Optional[Union[Text, List[Text]]] = None,
        not_intent: Optional[Union[Text, List[Text]]] = None,
        role: Optional[Text] = None,
        group: Optional[Text] = None,
    ) -> Dict[Text, Any]:
        """A dictionary for slot mapping to extract slot value.

        From:
        - an extracted entity
        - conditioned on
            - intent if it is not None
            - not_intent if it is not None,
                meaning user intent should not be this intent
            - role if it is not None
            - group if it is not None
        """
        intent, not_intent = (
            SlotMapping.to_list(intent),
            SlotMapping.to_list(not_intent),
        )

        return {
            "type": str(SlotMappingType.FROM_ENTITY),
            "entity": entity,
            "intent": intent,
            "not_intent": not_intent,
            "role": role,
            "group": group,
        }

    def get_mappings_for_slot(
        self, slot_to_fill: Text, domain: Domain
    ) -> List[Dict[Text, Any]]:
        """Get mappings for requested slot.

        If None, map requested slot to an entity with the same name
        """
        domain_slots = domain.as_dict().get(KEY_SLOTS, {})
        requested_slot_mappings = domain_slots.get(slot_to_fill, {}).get("mappings", [])

        # check provided slot mappings
        for requested_slot_mapping in requested_slot_mappings:
            if (
                not isinstance(requested_slot_mapping, dict)
                or requested_slot_mapping.get("type") is None
            ):
                raise TypeError("Provided incompatible slot mapping")

        return requested_slot_mappings

    def _create_unique_entity_mappings(self, domain: Domain) -> Set[Text]:
        """Finds mappings of type `from_entity` that uniquely set a slot.

        For example in the following form:
        some_form:
          departure_city:
            - type: from_entity
              entity: city
              role: from
            - type: from_entity
              entity: city
          arrival_city:
            - type: from_entity
              entity: city
              role: to
            - type: from_entity
              entity: city

        An entity `city` with a role `from` uniquely sets the slot `departure_city`
        and an entity `city` with a role `to` uniquely sets the slot `arrival_city`,
        so corresponding mappings are unique.
        But an entity `city` without a role can fill both `departure_city`
        and `arrival_city`, so corresponding mapping is not unique.

        Args:
            domain: The domain.

        Returns:
            A set of json dumps of unique mappings of type `from_entity`.
        """
        unique_entity_slot_mappings: Set[Text] = set()
        duplicate_entity_slot_mappings: Set[Text] = set()
        domain_slots = domain.as_dict().get(KEY_SLOTS, {})
        for slot in domain.required_slots_for_form(self.name()):
            for slot_mapping in domain_slots.get(slot, {}).get(SLOT_MAPPINGS, []):
                if slot_mapping.get(KEY_MAPPING_TYPE) == str(
                    SlotMappingType.FROM_ENTITY
                ):
                    mapping_as_string = json.dumps(slot_mapping, sort_keys=True)
                    if mapping_as_string in unique_entity_slot_mappings:
                        unique_entity_slot_mappings.remove(mapping_as_string)
                        duplicate_entity_slot_mappings.add(mapping_as_string)
                    elif mapping_as_string not in duplicate_entity_slot_mappings:
                        unique_entity_slot_mappings.add(mapping_as_string)

        return unique_entity_slot_mappings

    def entity_mapping_is_unique(
        self, slot_mapping: "SlotMapping", domain: Domain
    ) -> bool:
        """Verifies if the from_entity mapping is unique."""
        if not self._have_unique_entity_mappings_been_initialized:
            # create unique entity mappings on the first call
            self._unique_entity_mappings = self._create_unique_entity_mappings(domain)
            self._have_unique_entity_mappings_been_initialized = True

        mapping_as_string = json.dumps(slot_mapping.as_dict(), sort_keys=True)
        return mapping_as_string in self._unique_entity_mappings

    @staticmethod
    def get_entity_value_for_slot(
        name: Text,
        tracker: "DialogueStateTracker",
        slot_to_be_filled: Text,
        role: Optional[Text] = None,
        group: Optional[Text] = None,
    ) -> Any:
        """Extract entities for given name and optional role and group.

        Args:
            name: entity type (name) of interest
            tracker: the tracker
            slot_to_be_filled: Slot which is supposed to be filled by this entity.
            role: optional entity role of interest
            group: optional entity group of interest

        Returns:
            Value of entity.
        """
        # list is used to cover the case of list slot type
        value = list(
            tracker.get_latest_entity_values(name, entity_group=group, entity_role=role)
        )

        if isinstance(tracker.slots.get(slot_to_be_filled), ListSlot):
            return value

        if len(value) == 0:
            return None

        if len(value) == 1:
            return value[0]

        return value

    def get_slot_to_fill(self, tracker: "DialogueStateTracker") -> Optional[str]:
        """Gets the name of the slot which should be filled next.

        When switching to another form, the requested slot setting is still from the
        previous form and must be ignored.

        Returns:
            The slot name or `None`
        """
        return (
            tracker.get_slot(REQUESTED_SLOT)
            if tracker.active_loop_name == self.name()
            else None
        )

    async def validate_slots(
        self,
        slot_candidates: Dict[Text, Any],
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Union[SlotSet, Event]]:
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
            for the validated slots, if the custom form validation action is present in
            domain actions.
            Otherwise, returns empty list since the extracted slots already have
            corresponding `SlotSet` events in the tracker.
        """
        structlogger.debug(
            "forms.slots.validate", slot_candidates=copy.deepcopy(slot_candidates)
        )
        events: List[Union[SlotSet, Event]] = [
            SlotSet(slot_name, value) for slot_name, value in slot_candidates.items()
        ]

        validate_name = f"validate_{self.name()}"

        if validate_name not in domain.action_names_or_texts:
            return []

        # create temporary tracker with only the SlotSet events added
        # since last user utterance
        _tracker = self._temporary_tracker(tracker, events, domain)

        _action = RemoteAction(validate_name, self.action_endpoint)
        validate_events = await _action.run(output_channel, nlg, _tracker, domain)

        # Only return the validated SlotSet events by the custom form validation action
        # to avoid adding duplicate SlotSet events for slots that are already valid.
        return validate_events

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
            + [SlotSet(REQUESTED_SLOT, self.get_slot_to_fill(current_tracker))]
            # Insert form execution event so that it's clearly distinguishable which
            # events were newly added.
            + [ActionExecuted(self.name())]
            + additional_events,
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
            isinstance(event, ActionExecutionRejected) for event in validation_events
        )

    @staticmethod
    def _get_events_since_last_user_uttered(
        tracker: "DialogueStateTracker",
    ) -> List[SlotSet]:
        # TODO: Better way to get this latest_message index is through an instance
        # variable, eg. tracker.latest_message_index
        index_from_end = next(
            (
                i
                for i, event in enumerate(reversed(tracker.events))
                if event == Restarted() or event == tracker.latest_message
            ),
            len(tracker.events) - 1,
        )
        index = len(tracker.events) - index_from_end - 1
        events_since_last_user_uttered = [
            event
            for event in itertools.islice(tracker.events, index, None)
            if isinstance(event, SlotSet)
        ]

        return events_since_last_user_uttered

    def _update_slot_values(
        self,
        event: SlotSet,
        tracker: "DialogueStateTracker",
        domain: Domain,
        slot_values: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        slot_values[event.key] = event.value

        return slot_values

    def _add_dynamic_slots_requested_by_dynamic_forms(
        self, tracker: "DialogueStateTracker", domain: Domain
    ) -> Set[Text]:
        required_slots = set(self.required_slots(domain))
        requested_slot = self.get_slot_to_fill(tracker)

        if requested_slot:
            required_slots.add(requested_slot)

        return required_slots

    def _get_slot_extractions(
        self, tracker: "DialogueStateTracker", domain: Domain
    ) -> Dict[Text, Any]:
        events_since_last_user_uttered = FormAction._get_events_since_last_user_uttered(
            tracker
        )
        slot_values: Dict[Text, Any] = {}

        required_slots = self._add_dynamic_slots_requested_by_dynamic_forms(
            tracker, domain
        )

        for event in events_since_last_user_uttered:
            if event.key not in required_slots:
                continue

            slot_values = self._update_slot_values(event, tracker, domain, slot_values)

        return slot_values

    async def validate(
        self,
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Union[SlotSet, Event]]:
        """Extract and validate value of requested slot and other slots.

        Returns:
            The new validation events created by the custom form validation action

        Raises:
            ActionExecutionRejection exception to reject execution of form action
            if nothing was extracted.

        Subclass this method to add custom validation and rejection logic.
        """
        extracted_slot_values = self._get_slot_extractions(tracker, domain)

        validation_events = await self.validate_slots(
            extracted_slot_values, tracker, domain, output_channel, nlg
        )

        some_slots_were_validated = any(
            isinstance(event, SlotSet) and not event.key == REQUESTED_SLOT
            for event in validation_events
            # Ignore `SlotSet`s  for `REQUESTED_SLOT` as that's not a slot which needs
            # to be filled by the user.
        )

        # extract requested slot
        slot_to_fill = self.get_slot_to_fill(tracker)

        if (
            slot_to_fill
            and not extracted_slot_values
            and not some_slots_were_validated
            and not self._user_rejected_manually(validation_events)
        ):
            # reject to execute the form action
            # if some slot was requested but nothing was extracted
            # it will allow other policies to predict another action
            #
            # don't raise it here if the user rejected manually, to allow slots other
            # than the requested slot to be filled.
            #
            raise ActionExecutionRejection(
                self.name(),
                f"Failed to extract slot {slot_to_fill} with action {self.name()}",
            )
        return validation_events

    async def request_next_slot(
        self,
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        events_so_far: List[Event],
    ) -> List[Union[SlotSet, Event]]:
        """Request the next slot and response if needed, else return `None`."""
        request_slot_events: List[Event] = []

        if await self.is_done(output_channel, nlg, tracker, domain, events_so_far):
            # The custom action for slot validation decided to stop the form early
            return [SlotSet(REQUESTED_SLOT, None)]

        slot_to_request = next(
            (
                event.value
                for event in events_so_far
                if isinstance(event, SlotSet) and event.key == REQUESTED_SLOT
            ),
            None,
        )

        temp_tracker = self._temporary_tracker(tracker, events_so_far, domain)

        if not slot_to_request:
            slot_to_request = self._find_next_slot_to_request(temp_tracker, domain)
            request_slot_events.append(SlotSet(REQUESTED_SLOT, slot_to_request))

        if slot_to_request:
            bot_message_events = await self._ask_for_slot(
                domain, nlg, output_channel, slot_to_request, temp_tracker
            )
            return request_slot_events + bot_message_events

        # no more required slots to fill
        return [SlotSet(REQUESTED_SLOT, None)]

    def _find_next_slot_to_request(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Optional[Text]:
        return next(
            (
                slot
                for slot in self.required_slots(domain)
                if self._should_request_slot(tracker, slot)
            ),
            None,
        )

    def _name_of_utterance(self, domain: Domain, slot_name: Text) -> Optional[Text]:
        search_path = [
            f"action_ask_{self._form_name}_{slot_name}",
            f"{UTTER_PREFIX}ask_{self._form_name}_{slot_name}",
            f"action_ask_{slot_name}",
            f"{UTTER_PREFIX}ask_{slot_name}",
        ]

        found_actions = (
            action_name
            for action_name in search_path
            if action_name in domain.action_names_or_texts
        )

        return next(found_actions, None)

    async def _ask_for_slot(
        self,
        domain: Domain,
        nlg: NaturalLanguageGenerator,
        output_channel: OutputChannel,
        slot_name: Text,
        tracker: DialogueStateTracker,
    ) -> List[Event]:
        logger.debug(f"Request next slot '{slot_name}'")

        action_name_to_ask_for_next_slot = self._name_of_utterance(domain, slot_name)
        if not action_name_to_ask_for_next_slot:
            # Use a debug log as the user might have asked as part of a custom action
            logger.debug(
                f"There was no action found to ask for slot '{slot_name}' "
                f"name to be filled."
            )
            return []

        action_to_ask_for_next_slot = action.action_for_name_or_text(
            action_name_to_ask_for_next_slot, domain, self.action_endpoint
        )
        return await action_to_ask_for_next_slot.run(
            output_channel, nlg, tracker, domain
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
            and not tracker.is_active_loop_interrupted
        )

        if needs_validation:
            structlogger.debug(
                "forms.validation.required",
                tracker_latest_message=copy.deepcopy(tracker.latest_message),
            )
            return await self.validate(tracker, domain, output_channel, nlg)
        else:
            # Needed to determine which slots to request although there are no slots
            # to actually validate, which happens when coming back to the form after
            # an unhappy path
            return await self.validate_slots({}, tracker, domain, output_channel, nlg)

    @staticmethod
    def _should_request_slot(tracker: "DialogueStateTracker", slot_name: Text) -> bool:
        """Check whether form action should request given slot."""
        return tracker.get_slot(slot_name) is None

    async def activate(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Activate form if the form is called for the first time.

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
        logger.debug(f"Activated the form '{self.name()}'.")
        # collect values of required slots filled before activation
        prefilled_slots = {}

        action_extract_slots = action.action_for_name_or_text(
            ACTION_EXTRACT_SLOTS, domain, self.action_endpoint
        )

        logger.debug(
            f"Executing default action '{ACTION_EXTRACT_SLOTS}' at form activation."
        )

        extraction_events = await action_extract_slots.run(
            output_channel, nlg, tracker, domain, metadata
        )

        events_as_str = "\n".join(str(e) for e in extraction_events)
        logger.debug(
            f"The execution of '{ACTION_EXTRACT_SLOTS}' resulted in "
            f"these events: {events_as_str}."
        )

        tracker.update_with_events(extraction_events)

        for slot_name in self.required_slots(domain):
            if not self._should_request_slot(tracker, slot_name):
                prefilled_slots[slot_name] = tracker.get_slot(slot_name)

        if not prefilled_slots:
            logger.debug("No pre-filled required slots to validate.")
        else:
            structlogger.debug(
                "forms.validate.prefilled_slots",
                prefilled_slots=copy.deepcopy(prefilled_slots),
            )

        validate_name = f"validate_{self.name()}"

        if validate_name not in domain.action_names_or_texts:
            logger.debug(
                f"There is no validation action '{validate_name}' "
                f"to execute at form activation."
            )
            return [event for event in extraction_events if isinstance(event, SlotSet)]

        logger.debug(
            f"Executing validation action '{validate_name}' at form activation."
        )

        validated_events = await self.validate_slots(
            prefilled_slots, tracker, domain, output_channel, nlg
        )

        validated_slot_names = [
            event.key for event in validated_events if isinstance(event, SlotSet)
        ]

        return validated_events + [
            event
            for event in extraction_events
            if isinstance(event, SlotSet) and event.key not in validated_slot_names
        ]

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

        if not self._user_rejected_manually(events):
            events += await self.request_next_slot(
                tracker, domain, output_channel, nlg, events_so_far + events
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
        """Checks if loop can be terminated."""
        if any(isinstance(event, ActionExecutionRejected) for event in events_so_far):
            return False

        # Custom validation actions can decide to terminate the loop early by
        # setting the requested slot to `None` or setting `ActiveLoop(None)`.
        # We explicitly check only the last occurrences for each possible termination
        # event instead of doing `return event in events_so_far` to make it possible
        # to override termination events which were returned earlier.
        return next(
            (
                event
                for event in reversed(events_so_far)
                if isinstance(event, SlotSet) and event.key == REQUESTED_SLOT
            ),
            None,
        ) == SlotSet(REQUESTED_SLOT, None) or next(
            (
                event
                for event in reversed(events_so_far)
                if isinstance(event, ActiveLoop)
            ),
            None,
        ) == ActiveLoop(None)

    async def deactivate(self, *args: Any, **kwargs: Any) -> List[Event]:
        """Deactivates form."""
        logger.debug(f"Deactivating the form '{self.name()}'")
        return []

    async def _activate_loop(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        events = self._default_activation_events()

        temp_tracker = tracker.copy()
        temp_tracker.update_with_events(events)
        events += await self.activate(
            output_channel, nlg, temp_tracker, domain, metadata
        )

        return events
