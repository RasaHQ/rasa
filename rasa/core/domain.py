import collections
import copy
import json
import logging
import os
import typing
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Text, Tuple, Union

from ruamel.yaml import YAMLError

import rasa.core.constants
import rasa.shared.utils.io
from rasa.shared.core.domain import (
    BaseDomain,
    InvalidDomain,
    SessionConfig,
    CARRY_OVER_SLOTS_KEY,
    SESSION_EXPIRATION_TIME_KEY,
    SESSION_CONFIG_KEY,
    USED_ENTITIES_KEY,
    USE_ENTITIES_KEY,
    IGNORE_ENTITIES_KEY,
    KEY_SLOTS,
    KEY_FORMS,
    KEY_ACTIONS,
    KEY_ENTITIES,
    KEY_INTENTS,
    KEY_RESPONSES,
    KEY_E2E_ACTIONS,
)
import rasa.utils.io
from rasa.nlu.constants import INTENT_NAME_KEY
from rasa.utils.common import sort_list_of_dicts_by_first_key
from rasa.constants import (
    DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION,
    DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES,
)
from rasa.shared.constants import (
    DOCS_URL_DOMAINS
)
from rasa.core import utils
from rasa.core.actions import action  # pytype: disable=pyi-error
from rasa.core.actions.action import Action  # pytype: disable=pyi-error
from rasa.core.constants import (
    DEFAULT_KNOWLEDGE_BASE_ACTION,
    REQUESTED_SLOT,
    SLOT_LAST_OBJECT,
    SLOT_LAST_OBJECT_TYPE,
    SLOT_LISTED_ITEMS,
    USER,
    PREVIOUS_ACTION,
    ACTIVE_LOOP,
    SLOTS,
    SHOULD_NOT_BE_SET,
    LOOP_NAME,
)
from rasa.core.events import SlotSet, UserUttered
from rasa.utils.endpoints import EndpointConfig
from rasa.shared.core.slots import Slot, UnfeaturizedSlot, CategoricalSlot
from rasa.shared.nlu.constants import ENTITIES
from rasa.shared.utils.common import lazy_property

logger = logging.getLogger(__name__)

PREV_PREFIX = "prev_"

ALL_DOMAIN_KEYS = [
    KEY_SLOTS,
    KEY_FORMS,
    KEY_ACTIONS,
    KEY_ENTITIES,
    KEY_INTENTS,
    KEY_RESPONSES,
    KEY_E2E_ACTIONS,
]

# State is a dictionary with keys (USER, PREVIOUS_ACTION, SLOTS, ACTIVE_LOOP)
# representing the origin of a SubState;
# the values are SubStates, that contain the information needed for featurization
SubState = Dict[Text, Union[Text, Tuple[Union[float, Text]]]]
State = Dict[Text, SubState]

if typing.TYPE_CHECKING:
    from rasa.core.trackers import DialogueStateTracker


class Domain(BaseDomain):
    """The domain specifies the universe in which the bot's policy acts.

    A Domain subclass provides the actions the bot can take, the intents
    and entities it can recognise."""

    @classmethod
    def load(cls, paths: Union[List[Text], Text]) -> "Domain":
        if not paths:
            raise InvalidDomain(
                "No domain file was specified. Please specify a path "
                "to a valid domain file."
            )
        elif not isinstance(paths, list) and not isinstance(paths, set):
            paths = [paths]

        domain = Domain.empty()
        for path in paths:
            other = cls.from_path(path)
            domain = domain.merge(other)

        return domain

    @classmethod
    def from_path(cls, path: Text) -> "Domain":
        path = os.path.abspath(path)

        if os.path.isfile(path):
            domain = cls.from_file(path)
        elif os.path.isdir(path):
            domain = cls.from_directory(path)
        else:
            raise InvalidDomain(
                "Failed to load domain specification from '{}'. "
                "File not found!".format(os.path.abspath(path))
            )

        return domain

    @classmethod
    def from_file(cls, path: Text) -> "Domain":
        return cls.from_yaml(rasa.shared.utils.io.read_file(path), path)

    @staticmethod
    def _get_session_config(session_config: Dict) -> SessionConfig:
        session_expiration_time_min = session_config.get(SESSION_EXPIRATION_TIME_KEY)

        # TODO: 2.0 reconsider how to apply sessions to old projects and legacy trackers
        if session_expiration_time_min is None:
            session_expiration_time_min = DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES

        carry_over_slots = session_config.get(
            CARRY_OVER_SLOTS_KEY, DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION
        )

        return SessionConfig(session_expiration_time_min, carry_over_slots)

    @classmethod
    def from_directory(cls, path: Text) -> "Domain":
        """Loads and merges multiple domain files recursively from a directory tree."""

        domain = Domain.empty()
        for root, _, files in os.walk(path, followlinks=True):
            for file in files:
                full_path = os.path.join(root, file)
                if Domain.is_domain_file(full_path):
                    other = Domain.from_file(full_path)
                    domain = other.merge(domain)

        return domain

    def merge(self, domain: Optional["Domain"], override: bool = False) -> "Domain":
        """Merge this domain with another one, combining their attributes.

        List attributes like ``intents`` and ``actions`` will be deduped
        and merged. Single attributes will be taken from `self` unless
        override is `True`, in which case they are taken from `domain`."""

        if not domain or domain.is_empty():
            return self

        if self.is_empty():
            return domain

        domain_dict = domain.as_dict()
        combined = self.as_dict()

        def merge_dicts(
            d1: Dict[Text, Any],
            d2: Dict[Text, Any],
            override_existing_values: bool = False,
        ) -> Dict[Text, Any]:
            if override_existing_values:
                a, b = d1.copy(), d2.copy()
            else:
                a, b = d2.copy(), d1.copy()
            a.update(b)
            return a

        def merge_lists(l1: List[Any], l2: List[Any]) -> List[Any]:
            return sorted(list(set(l1 + l2)))

        def merge_lists_of_dicts(
            dict_list1: List[Dict],
            dict_list2: List[Dict],
            override_existing_values: bool = False,
        ) -> List[Dict]:
            dict1 = {list(i.keys())[0]: i for i in dict_list1}
            dict2 = {list(i.keys())[0]: i for i in dict_list2}
            merged_dicts = merge_dicts(dict1, dict2, override_existing_values)
            return list(merged_dicts.values())

        if override:
            config = domain_dict["config"]
            for key, val in config.items():  # pytype: disable=attribute-error
                combined["config"][key] = val

        if override or self.session_config == SessionConfig.default():
            combined[SESSION_CONFIG_KEY] = domain_dict[SESSION_CONFIG_KEY]

        combined[KEY_INTENTS] = merge_lists_of_dicts(
            combined[KEY_INTENTS], domain_dict[KEY_INTENTS], override
        )
        combined[KEY_FORMS] = merge_lists_of_dicts(
            combined[KEY_FORMS], domain_dict[KEY_FORMS], override
        )

        # remove existing forms from new actions
        for form in combined[KEY_FORMS]:
            if form in domain_dict[KEY_ACTIONS]:
                domain_dict[KEY_ACTIONS].remove(form)

        for key in [KEY_ENTITIES, KEY_ACTIONS, KEY_E2E_ACTIONS]:
            combined[key] = merge_lists(combined[key], domain_dict[key])

        for key in [KEY_RESPONSES, KEY_SLOTS]:
            combined[key] = merge_dicts(combined[key], domain_dict[key], override)

        return self.__class__.from_dict(combined)

    @staticmethod
    def _transform_intent_properties_for_internal_use(
        intent: Dict[Text, Any], entities: List
    ) -> Dict[Text, Any]:
        """Transform intent properties coming from a domain file for internal use.

        In domain files, `use_entities` or `ignore_entities` is used. Internally, there
        is a property `used_entities` instead that lists all entities to be used.

        Args:
            intent: The intents as provided by a domain file.
            entities: All entities as provided by a domain file.

        Returns:
            The intents as they should be used internally.
        """
        name, properties = list(intent.items())[0]

        properties.setdefault(USE_ENTITIES_KEY, True)
        properties.setdefault(IGNORE_ENTITIES_KEY, [])
        if not properties[USE_ENTITIES_KEY]:  # this covers False, None and []
            properties[USE_ENTITIES_KEY] = []

        # `use_entities` is either a list of explicitly included entities
        # or `True` if all should be included
        if properties[USE_ENTITIES_KEY] is True:
            included_entities = set(entities)
        else:
            included_entities = set(properties[USE_ENTITIES_KEY])
        excluded_entities = set(properties[IGNORE_ENTITIES_KEY])
        used_entities = list(included_entities - excluded_entities)
        used_entities.sort()

        # Only print warning for ambiguous configurations if entities were included
        # explicitly.
        explicitly_included = isinstance(properties[USE_ENTITIES_KEY], list)
        ambiguous_entities = included_entities.intersection(excluded_entities)
        if explicitly_included and ambiguous_entities:
            rasa.shared.utils.io.raise_warning(
                f"Entities: '{ambiguous_entities}' are explicitly included and"
                f" excluded for intent '{name}'."
                f"Excluding takes precedence in this case. "
                f"Please resolve that ambiguity.",
                docs=f"{DOCS_URL_DOMAINS}#ignoring-entities-for-certain-intents",
            )

        properties[USED_ENTITIES_KEY] = used_entities
        del properties[USE_ENTITIES_KEY]
        del properties[IGNORE_ENTITIES_KEY]

        return intent

    def __hash__(self) -> int:
        self_as_dict = self.as_dict()
        self_as_dict[KEY_INTENTS] = sort_list_of_dicts_by_first_key(
            self_as_dict[KEY_INTENTS]
        )
        self_as_dict[KEY_ACTIONS] = self.action_names
        self_as_string = json.dumps(self_as_dict, sort_keys=True)
        text_hash = rasa.shared.utils.io.get_text_hash(self_as_string)

        return int(text_hash, 16)

    @lazy_property
    def num_actions(self):
        """Returns the number of available actions."""

        # noinspection PyTypeChecker
        return len(self.action_names)

    @lazy_property
    def num_states(self):
        """Number of used input states for the action prediction."""

        return len(self.input_states)

    def add_categorical_slot_default_value(self) -> None:
        """Add a default value to all categorical slots.

        All unseen values found for the slot will be mapped to this default value
        for featurization.
        """
        for slot in [s for s in self.slots if type(s) is CategoricalSlot]:
            slot.add_default_value()

    def add_requested_slot(self) -> None:
        """Add a slot called `requested_slot` to the list of slots.

        The value of this slot will hold the name of the slot which the user
        needs to fill in next (either explicitly or implicitly) as part of a form.
        """
        if self.form_names and REQUESTED_SLOT not in [s.name for s in self.slots]:
            self.slots.append(UnfeaturizedSlot(REQUESTED_SLOT))

    def add_knowledge_base_slots(self) -> None:
        """
        Add slots for the knowledge base action to the list of slots, if the
        default knowledge base action name is present.

        As soon as the knowledge base action is not experimental anymore, we should
        consider creating a new section in the domain file dedicated to knowledge
        base slots.
        """
        if DEFAULT_KNOWLEDGE_BASE_ACTION in self.action_names:
            logger.warning(
                "You are using an experiential feature: Action '{}'!".format(
                    DEFAULT_KNOWLEDGE_BASE_ACTION
                )
            )
            slot_names = [s.name for s in self.slots]
            knowledge_base_slots = [
                SLOT_LISTED_ITEMS,
                SLOT_LAST_OBJECT,
                SLOT_LAST_OBJECT_TYPE,
            ]
            for s in knowledge_base_slots:
                if s not in slot_names:
                    self.slots.append(UnfeaturizedSlot(s))

    def action_for_name(
        self, action_name: Text, action_endpoint: Optional[EndpointConfig]
    ) -> Optional[Action]:
        """Look up which action corresponds to this action name."""

        if action_name not in self.action_names:
            self._raise_action_not_found_exception(action_name)

        should_use_form_action = (
            action_name in self.form_names and self.slot_mapping_for_form(action_name)
        )

        return action.action_from_name(
            action_name,
            action_endpoint,
            self.user_actions_and_forms,
            should_use_form_action,
        )

    def action_for_index(
        self, index: int, action_endpoint: Optional[EndpointConfig]
    ) -> Optional[Action]:
        """Integer index corresponding to an actions index in the action list.

        This method resolves the index to the actions name."""

        if self.num_actions <= index or index < 0:
            raise IndexError(
                "Cannot access action at index {}. "
                "Domain has {} actions."
                "".format(index, self.num_actions)
            )

        return self.action_for_name(self.action_names[index], action_endpoint)

    def actions(self, action_endpoint) -> List[Optional[Action]]:
        return [
            self.action_for_name(name, action_endpoint) for name in self.action_names
        ]

    def index_for_action(self, action_name: Text) -> Optional[int]:
        """Look up which action index corresponds to this action name."""

        try:
            return self.action_names.index(action_name)
        except ValueError:
            self._raise_action_not_found_exception(action_name)

    def _raise_action_not_found_exception(self, action_name) -> typing.NoReturn:
        action_names = "\n".join([f"\t - {a}" for a in self.action_names])
        raise NameError(
            f"Cannot access action '{action_name}', "
            f"as that name is not a registered "
            f"action for this domain. "
            f"Available actions are: \n{action_names}"
        )

    def random_template_for(self, utter_action: Text) -> Optional[Dict[Text, Any]]:
        import numpy as np

        if utter_action in self.templates:
            return np.random.choice(self.templates[utter_action])
        else:
            return None

    # noinspection PyTypeChecker
    @lazy_property
    def slot_states(self) -> List[Text]:
        """Returns all available slot state strings."""

        return [
            f"{slot.name}_{feature_index}"
            for slot in self.slots
            for feature_index in range(0, slot.feature_dimensionality())
        ]

    @lazy_property
    def input_state_map(self) -> Dict[Text, int]:
        """Provide a mapping from state names to indices."""
        return {f: i for i, f in enumerate(self.input_states)}

    @lazy_property
    def input_states(self) -> List[Text]:
        """Returns all available states."""

        return (
            self.intents
            + self.entities
            + self.slot_states
            + self.action_names
            + self.form_names
        )

    def _get_featurized_entities(self, latest_message: UserUttered) -> Set[Text]:
        intent_name = latest_message.intent.get(INTENT_NAME_KEY)
        intent_config = self.intent_config(intent_name)
        entities = latest_message.entities
        entity_names = set(
            entity["entity"] for entity in entities if "entity" in entity.keys()
        )

        wanted_entities = set(intent_config.get(USED_ENTITIES_KEY, entity_names))

        return entity_names.intersection(wanted_entities)

    def _get_user_sub_state(
        self, tracker: "DialogueStateTracker"
    ) -> Dict[Text, Union[Text, Tuple[Text]]]:
        """Turn latest UserUttered event into a substate containing intent,
        text and set entities if present
        Args:
            tracker: dialog state tracker containing the dialog so far
        Returns:
            a dictionary containing intent, text and set entities
        """
        latest_message = tracker.latest_message
        if not latest_message or latest_message.is_empty():
            return {}

        sub_state = latest_message.as_sub_state()

        # filter entities based on intent config
        # sub_state will be transformed to frozenset therefore we need to
        # convert the list to the tuple
        # sub_state is transformed to frozenset because we will later hash it
        # for deduplication
        entities = tuple(self._get_featurized_entities(latest_message))
        if entities:
            sub_state[ENTITIES] = entities
        else:
            sub_state.pop(ENTITIES, None)

        return sub_state

    @staticmethod
    def _get_slots_sub_state(
        tracker: "DialogueStateTracker",
    ) -> Dict[Text, Union[Text, Tuple[float]]]:
        """Set all set slots with the featurization of the stored value
        Args:
            tracker: dialog state tracker containing the dialog so far
        Returns:
            a dictionary mapping slot names to their featurization
        """

        # proceed with values only if the user of a bot have done something
        # at the previous step i.e., when the state is not empty.
        if tracker.latest_message == UserUttered.empty() or not tracker.latest_action:
            return {}

        slots = {}
        for slot_name, slot in tracker.slots.items():
            if slot is not None and slot.as_feature():
                if slot.value == SHOULD_NOT_BE_SET:
                    slots[slot_name] = SHOULD_NOT_BE_SET
                elif any(slot.as_feature()):
                    # only add slot if some of the features are not zero
                    slots[slot_name] = tuple(slot.as_feature())

        return slots

    @staticmethod
    def _get_prev_action_sub_state(
        tracker: "DialogueStateTracker",
    ) -> Dict[Text, Text]:
        """Turn the previous taken action into a state name.
        Args:
            tracker: dialog state tracker containing the dialog so far
        Returns:
            a dictionary with the information on latest action
        """
        return tracker.latest_action

    @staticmethod
    def _get_active_loop_sub_state(
        tracker: "DialogueStateTracker",
    ) -> Dict[Text, Text]:
        """Turn tracker's active loop into a state name.
        Args:
            tracker: dialog state tracker containing the dialog so far
        Returns:
            a dictionary mapping "name" to active loop name if present
        """

        # we don't use tracker.active_loop_name
        # because we need to keep should_not_be_set
        active_loop = tracker.active_loop.get(LOOP_NAME)
        if active_loop:
            return {LOOP_NAME: active_loop}
        else:
            return {}

    @staticmethod
    def _clean_state(state: State) -> State:
        return {
            state_type: sub_state
            for state_type, sub_state in state.items()
            if sub_state
        }

    def get_active_states(self, tracker: "DialogueStateTracker") -> State:
        """Return a bag of active states from the tracker state."""
        state = {
            USER: self._get_user_sub_state(tracker),
            SLOTS: self._get_slots_sub_state(tracker),
            PREVIOUS_ACTION: self._get_prev_action_sub_state(tracker),
            ACTIVE_LOOP: self._get_active_loop_sub_state(tracker),
        }
        return self._clean_state(state)

    def states_for_tracker_history(
        self, tracker: "DialogueStateTracker"
    ) -> List[State]:
        """Array of states for each state of the trackers history."""
        return [
            self.get_active_states(tr) for tr in tracker.generate_all_prior_trackers()
        ]

    def slots_for_entities(self, entities: List[Dict[Text, Any]]) -> List[SlotSet]:
        if self.store_entities_as_slots:
            slot_events = []
            for s in self.slots:
                if s.auto_fill:
                    matching_entities = [
                        e["value"] for e in entities if e["entity"] == s.name
                    ]
                    if matching_entities:
                        if s.type_name == "list":
                            slot_events.append(SlotSet(s.name, matching_entities))
                        else:
                            slot_events.append(SlotSet(s.name, matching_entities[-1]))
            return slot_events
        else:
            return []

    def persist_specification(self, model_path: Text) -> None:
        """Persist the domain specification to storage."""

        domain_spec_path = os.path.join(model_path, "domain.json")
        rasa.utils.io.create_directory_for_file(domain_spec_path)

        metadata = {"states": self.input_states}
        rasa.utils.io.dump_obj_as_json_to_file(domain_spec_path, metadata)

    @classmethod
    def load_specification(cls, path: Text) -> Dict[Text, Any]:
        """Load a domains specification from a dumped model directory."""

        metadata_path = os.path.join(path, "domain.json")
        specification = json.loads(rasa.shared.utils.io.read_file(metadata_path))
        return specification

    def compare_with_specification(self, path: Text) -> bool:
        """Compare the domain spec of the current and the loaded domain.

        Throws exception if the loaded domain specification is different
        to the current domain are different."""

        loaded_domain_spec = self.load_specification(path)
        states = loaded_domain_spec["states"]

        if set(states) != set(self.input_states):
            missing = ",".join(set(states) - set(self.input_states))
            additional = ",".join(set(self.input_states) - set(states))
            raise InvalidDomain(
                f"Domain specification has changed. "
                f"You MUST retrain the policy. "
                f"Detected mismatch in domain specification. "
                f"The following states have been \n"
                f"\t - removed: {missing} \n"
                f"\t - added:   {additional} "
            )
        else:
            return True

    def persist(self, filename: Union[Text, Path]) -> None:
        """Write domain to a file."""

        domain_data = self.as_dict()
        utils.dump_obj_as_yaml_to_file(
            filename, domain_data, should_preserve_key_order=True
        )

    def persist_clean(self, filename: Text) -> None:
        """Write cleaned domain to a file."""

        cleaned_domain_data = self.cleaned_domain()
        utils.dump_obj_as_yaml_to_file(
            filename, cleaned_domain_data, should_preserve_key_order=True
        )

    def as_yaml(self, clean_before_dump: bool = False) -> Text:
        if clean_before_dump:
            domain_data = self.cleaned_domain()
        else:
            domain_data = self.as_dict()

        return utils.dump_obj_as_yaml_to_string(domain_data)

    def intent_config(self, intent_name: Text) -> Dict[Text, Any]:
        """Return the configuration for an intent."""
        return self.intent_properties.get(intent_name, {})

    def check_missing_templates(self) -> None:
        """Warn user of utterance names which have no specified template."""

        utterances = [
            a
            for a in self.action_names
            if a.startswith(rasa.core.constants.UTTER_PREFIX)
        ]

        missing_templates = [t for t in utterances if t not in self.templates.keys()]

        if missing_templates:
            for template in missing_templates:
                rasa.shared.utils.io.raise_warning(
                    f"Action '{template}' is listed as a "
                    f"response action in the domain file, but there is "
                    f"no matching response defined. Please "
                    f"check your domain.",
                    docs=DOCS_URL_DOMAINS + "#responses",
                )

    @staticmethod
    def is_domain_file(filename: Text) -> bool:
        """Checks whether the given file path is a Rasa domain file.

        Args:
            filename: Path of the file which should be checked.

        Returns:
            `True` if it's a domain file, otherwise `False`.
        """
        from rasa.shared.data import is_likely_yaml_file

        if not is_likely_yaml_file(filename):
            return False
        try:
            content = rasa.shared.utils.io.read_yaml_file(filename)
            if any(key in content for key in ALL_DOMAIN_KEYS):
                return True
        except YAMLError:
            pass

        return False

    def slot_mapping_for_form(self, form_name: Text) -> Dict:
        """Retrieve the slot mappings for a form which are defined in the domain.

        Options:
        - an extracted entity
        - intent: value pairs
        - trigger_intent: value pairs
        - a whole message
        or a list of them, where the first match will be picked

        Args:
            form_name: The name of the form.

        Returns:
            The slot mapping or an empty dictionary in case no mapping was found.
        """
        return next(
            (form[form_name] for form in self.forms if form_name in form.keys()), {}
        )


class TemplateDomain(Domain):
    pass
