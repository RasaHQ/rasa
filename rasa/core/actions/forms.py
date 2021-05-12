from typing import Text, List, Optional, Union, Any, Dict, Tuple, Set
import logging
import json

from rasa.core.actions import action
from rasa.core.actions.loops import LoopAction
from rasa.core.channels import OutputChannel
from rasa.shared.core.domain import Domain, InvalidDomain, SlotMapping

from rasa.core.actions.action import ActionExecutionRejection, RemoteAction
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    REQUESTED_SLOT,
    LOOP_INTERRUPTED,
)
from rasa.shared.constants import (
    UTTER_PREFIX,
    IGNORED_INTENTS,
)
from rasa.shared.core.events import (
    Event,
    SlotSet,
    ActionExecuted,
    ActiveLoop,
    ActionExecutionRejected,
)
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.shared.core.slots import ListSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


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
        self._unique_entity_mappings = None

    def name(self) -> Text:
        return self._form_name

    def required_slots(self, domain: Domain) -> List[Text]:
        """A list of required slots that the form has to fill.

        Returns:
            A list of slot names.
        """
        return list(domain.slot_mapping_for_form(self.name()).keys())

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
        intent, not_intent = self._list_intents(intent, not_intent)

        return {
            "type": str(SlotMapping.FROM_ENTITY),
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
        requested_slot_mappings = self._to_list(
            domain.slot_mapping_for_form(self.name()).get(
                slot_to_fill, self.from_entity(slot_to_fill),
            )
        )
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
        unique_entity_slot_mappings = set()
        duplicate_entity_slot_mappings = set()
        for slot_mappings in domain.slot_mapping_for_form(self.name()).values():
            for slot_mapping in slot_mappings:
                if slot_mapping.get("type") == str(SlotMapping.FROM_ENTITY):
                    mapping_as_string = json.dumps(slot_mapping, sort_keys=True)
                    if mapping_as_string in unique_entity_slot_mappings:
                        unique_entity_slot_mappings.remove(mapping_as_string)
                        duplicate_entity_slot_mappings.add(mapping_as_string)
                    elif mapping_as_string not in duplicate_entity_slot_mappings:
                        unique_entity_slot_mappings.add(mapping_as_string)

        return unique_entity_slot_mappings

    def _entity_mapping_is_unique(
        self, slot_mapping: Dict[Text, Any], domain: Domain
    ) -> bool:
        if self._unique_entity_mappings is None:
            # create unique entity mappings on the first call
            self._unique_entity_mappings = self._create_unique_entity_mappings(domain)

        mapping_as_string = json.dumps(slot_mapping, sort_keys=True)
        return mapping_as_string in self._unique_entity_mappings

    def get_ignored_intents(self, domain: Domain) -> List[Text]:
        """Returns a list of ignored intents.

        Args:
            domain: The current model domain.

        Returns:
            The value/s found in `ignored_intents` parameter in the `domain.yml`
            (under forms).
        """
        ignored_intents = domain.forms[self.name()].get(IGNORED_INTENTS, [])
        if not isinstance(ignored_intents, list):
            ignored_intents = [ignored_intents]

        return ignored_intents

    def intent_is_desired(
        self,
        requested_slot_mapping: Dict[Text, Any],
        tracker: "DialogueStateTracker",
        domain: Domain,
    ) -> bool:
        """Check whether user intent matches intent conditions."""
        mapping_intents = FormAction._to_list(requested_slot_mapping.get("intent", []))
        mapping_not_intents = FormAction._to_list(
            requested_slot_mapping.get("not_intent", [])
        )

        mapping_not_intents = set(
            mapping_not_intents + self.get_ignored_intents(domain)
        )

        intent = tracker.latest_message.intent.get("name")

        intent_not_blocked = not mapping_intents and intent not in mapping_not_intents

        return intent_not_blocked or intent in mapping_intents

    def entity_is_desired(
        self,
        slot_mapping: Dict[Text, Any],
        slot: Text,
        entity_type_of_slot_to_fill: Optional[Text],
        tracker: DialogueStateTracker,
        domain: Domain,
    ) -> bool:
        """Check whether slot should be filled by an entity in the input or not.

        Args:
            slot_mapping: Slot mapping.
            slot: The slot to be filled.
            entity_type_of_slot_to_fill: Entity type of slot to fill.
            tracker: The tracker.
            domain: The domain.

        Returns:
            True, if slot should be filled, false otherwise.
        """
        # slot name is equal to the entity type
        slot_equals_entity = slot == slot_mapping.get("entity")
        # if entity mapping is unique, it means that an entity always sets
        # a certain slot, so try to extract this slot if entity matches slot mapping
        entity_mapping_is_unique = self._entity_mapping_is_unique(slot_mapping, domain)

        # use the custom slot mapping 'from_entity' defined by the user to check
        # whether we can fill a slot with an entity (only if a role or a group label
        # is set)
        if (
            slot_mapping.get("role") is None and slot_mapping.get("group") is None
        ) or entity_type_of_slot_to_fill != slot_mapping.get("entity"):
            slot_fulfils_entity_mapping = False
        else:
            matching_values = self.get_entity_value_for_slot(
                slot_mapping.get("entity"),
                tracker,
                slot,
                slot_mapping.get("role"),
                slot_mapping.get("group"),
            )
            slot_fulfils_entity_mapping = matching_values is not None

        return (
            slot_equals_entity
            or entity_mapping_is_unique
            or slot_fulfils_entity_mapping
        )

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

    def extract_other_slots(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Dict[Text, Any]:
        """Extract the values of the other slots
        if they are set by corresponding entities from the user input
        else return `None`.
        """
        slot_to_fill = self.get_slot_to_fill(tracker)

        entity_type_of_slot_to_fill = self._get_entity_type_of_slot_to_fill(
            slot_to_fill, domain
        )

        slot_values = {}
        for slot in self.required_slots(domain):
            # look for other slots
            if slot != slot_to_fill:
                # list is used to cover the case of list slot type
                slot_mappings = self.get_mappings_for_slot(slot, domain)

                for slot_mapping in slot_mappings:
                    # check whether the slot should be filled by an entity in the input
                    should_fill_entity_slot = (
                        slot_mapping["type"] == str(SlotMapping.FROM_ENTITY)
                        and self.intent_is_desired(slot_mapping, tracker, domain)
                        and self.entity_is_desired(
                            slot_mapping,
                            slot,
                            entity_type_of_slot_to_fill,
                            tracker,
                            domain,
                        )
                    )
                    # check whether the slot should be
                    # filled from trigger intent mapping
                    should_fill_trigger_slot = (
                        tracker.active_loop_name != self.name()
                        and slot_mapping["type"] == str(SlotMapping.FROM_TRIGGER_INTENT)
                        and self.intent_is_desired(slot_mapping, tracker, domain)
                    )
                    if should_fill_entity_slot:
                        value = self.get_entity_value_for_slot(
                            slot_mapping["entity"],
                            tracker,
                            slot,
                            slot_mapping.get("role"),
                            slot_mapping.get("group"),
                        )
                    elif should_fill_trigger_slot:
                        value = slot_mapping.get("value")
                    else:
                        value = None

                    if value is not None:
                        logger.debug(f"Extracted '{value}' for extra slot '{slot}'.")
                        slot_values[slot] = value
                        # this slot is done, check  next
                        break

        return slot_values

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

    def extract_requested_slot(
        self, tracker: "DialogueStateTracker", domain: Domain, slot_to_fill: Text,
    ) -> Dict[Text, Any]:
        """Extract the value of requested slot from a user input else return `None`.

        Args:
            tracker: a DialogueStateTracker instance
            domain: the current domain
            slot_to_fill: the name of the slot to fill

        Returns:
            a dictionary with one key being the name of the slot to fill
            and its value being the slot value, or an empty dictionary
            if no slot value was found.
        """
        logger.debug(f"Trying to extract requested slot '{slot_to_fill}' ...")

        # get mapping for requested slot
        requested_slot_mappings = self.get_mappings_for_slot(slot_to_fill, domain)
        for requested_slot_mapping in requested_slot_mappings:
            logger.debug(f"Got mapping '{requested_slot_mapping}'")

            if self.intent_is_desired(requested_slot_mapping, tracker, domain):
                mapping_type = requested_slot_mapping["type"]
                if mapping_type == str(SlotMapping.FROM_ENTITY):
                    value = self.get_entity_value_for_slot(
                        requested_slot_mapping.get("entity"),
                        tracker,
                        slot_to_fill,
                        requested_slot_mapping.get("role"),
                        requested_slot_mapping.get("group"),
                    )
                elif mapping_type == str(SlotMapping.FROM_INTENT):
                    value = requested_slot_mapping.get("value")
                elif mapping_type == str(SlotMapping.FROM_TRIGGER_INTENT):
                    # from_trigger_intent is only used on form activation
                    continue
                elif mapping_type == str(SlotMapping.FROM_TEXT):
                    value = tracker.latest_message.text
                else:
                    raise InvalidDomain("Provided slot mapping type is not supported")

                if value is not None:
                    logger.debug(
                        f"Successfully extracted '{value}' for requested slot "
                        f"'{slot_to_fill}'"
                    )
                    return {slot_to_fill: value}

        logger.debug(f"Failed to extract requested slot '{slot_to_fill}'")
        return {}

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
            SlotSet(slot_name, value) for slot_name, value in slot_candidates.items()
        ]

        validate_name = f"validate_{self.name()}"

        if validate_name not in domain.action_names_or_texts:
            return events

        _tracker = self._temporary_tracker(tracker, events, domain)
        _action = RemoteAction(validate_name, self.action_endpoint)
        validate_events = await _action.run(output_channel, nlg, _tracker, domain)

        validated_slot_names = [
            event.key for event in validate_events if isinstance(event, SlotSet)
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
            + [SlotSet(REQUESTED_SLOT, self.get_slot_to_fill(current_tracker))]
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
            isinstance(event, ActionExecutionRejected) for event in validation_events
        )

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
        # extract other slots that were not requested
        # but set by corresponding entity or trigger intent mapping
        slot_values = self.extract_other_slots(tracker, domain)

        # extract requested slot
        slot_to_fill = self.get_slot_to_fill(tracker)
        if slot_to_fill:
            slot_values.update(
                self.extract_requested_slot(tracker, domain, slot_to_fill)
            )

        validation_events = await self.validate_slots(
            slot_values, tracker, domain, output_channel, nlg
        )

        some_slots_were_validated = any(
            isinstance(event, SlotSet)
            for event in validation_events
            # Ignore `SlotSet`s  for `REQUESTED_SLOT` as that's not a slot which needs
            # to be filled by the user.
            if isinstance(event, SlotSet) and not event.key == REQUESTED_SLOT
        )

        if (
            slot_to_fill
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
    ) -> List[Event]:
        """Request the next slot and response if needed, else return `None`."""
        request_slot_events = []

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

        action_to_ask_for_next_slot = self._name_of_utterance(domain, slot_name)
        if not action_to_ask_for_next_slot:
            # Use a debug log as the user might have asked as part of a custom action
            logger.debug(
                f"There was no action found to ask for slot '{slot_name}' "
                f"name to be filled."
            )
            return []

        action_to_ask_for_next_slot = action.action_for_name_or_text(
            action_to_ask_for_next_slot, domain, self.action_endpoint
        )
        return await action_to_ask_for_next_slot.run(
            output_channel, nlg, tracker, domain
        )

    # helpers
    @staticmethod
    def _to_list(x: Optional[Any]) -> List[Any]:
        """Convert object to a list if it isn't."""
        if x is None:
            x = []
        elif not isinstance(x, list):
            x = [x]

        return x

    def _list_intents(
        self,
        intent: Optional[Union[Text, List[Text]]] = None,
        not_intent: Optional[Union[Text, List[Text]]] = None,
    ) -> Tuple[List[Text], List[Text]]:
        """Check provided intent and not_intent."""
        if intent and not_intent:
            raise ValueError(
                f"Providing  both intent '{intent}' and not_intent '{not_intent}' "
                f"is not supported."
            )

        return self._to_list(intent), self._to_list(not_intent)

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

        logger.debug(f"Validating pre-filled required slots: {prefilled_slots}")
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
        events = await self._validate_if_required(tracker, domain, output_channel, nlg)

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
        ) == ActiveLoop(
            None
        )

    async def deactivate(self, *args: Any, **kwargs: Any) -> List[Event]:
        logger.debug(f"Deactivating the form '{self.name()}'")
        return []

    def _get_entity_type_of_slot_to_fill(
        self, slot_to_fill: Text, domain: "Domain"
    ) -> Optional[Text]:
        if not slot_to_fill:
            return None

        mappings = self.get_mappings_for_slot(slot_to_fill, domain)
        mappings = [
            m for m in mappings if m.get("type") == str(SlotMapping.FROM_ENTITY)
        ]

        if not mappings:
            return None

        entity_type = mappings[0].get("entity")

        for i in range(1, len(mappings)):
            if entity_type != mappings[i].get("entity"):
                return None

        return entity_type
