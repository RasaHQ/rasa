import copy
import collections
import json
import logging
import os
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Text,
    Tuple,
    Union,
    NoReturn,
    TYPE_CHECKING,
)
from pathlib import Path

import rasa.shared.constants
import rasa.shared.core.constants
from rasa.shared.exceptions import RasaException, YamlException
from rasa.shared.utils.validation import YamlValidationException
import rasa.shared.nlu.constants
import rasa.shared.utils.validation
import rasa.shared.utils.io
import rasa.shared.utils.common
from rasa.shared.core.events import SlotSet, UserUttered
from rasa.shared.core.slots import Slot, CategoricalSlot, TextSlot
from rasa.shared.utils.validation import KEY_TRAINING_DATA_FORMAT_VERSION


if TYPE_CHECKING:
    from rasa.shared.core.trackers import DialogueStateTracker

CARRY_OVER_SLOTS_KEY = "carry_over_slots_to_new_session"
SESSION_EXPIRATION_TIME_KEY = "session_expiration_time"
SESSION_CONFIG_KEY = "session_config"
USED_ENTITIES_KEY = "used_entities"
USE_ENTITIES_KEY = "use_entities"
IGNORE_ENTITIES_KEY = "ignore_entities"
IS_RETRIEVAL_INTENT_KEY = "is_retrieval_intent"
ENTITY_ROLES_KEY = "roles"
ENTITY_GROUPS_KEY = "groups"

KEY_SLOTS = "slots"
KEY_INTENTS = "intents"
KEY_ENTITIES = "entities"
KEY_RESPONSES = "responses"
KEY_ACTIONS = "actions"
KEY_FORMS = "forms"
KEY_E2E_ACTIONS = "e2e_actions"

ALL_DOMAIN_KEYS = [
    KEY_SLOTS,
    KEY_FORMS,
    KEY_ACTIONS,
    KEY_ENTITIES,
    KEY_INTENTS,
    KEY_RESPONSES,
    KEY_E2E_ACTIONS,
]

PREV_PREFIX = "prev_"

# State is a dictionary with keys (USER, PREVIOUS_ACTION, SLOTS, ACTIVE_LOOP)
# representing the origin of a SubState;
# the values are SubStates, that contain the information needed for featurization
SubState = Dict[Text, Union[Text, Tuple[Union[float, Text]]]]
State = Dict[Text, SubState]

logger = logging.getLogger(__name__)


class InvalidDomain(RasaException):
    """Exception that can be raised when domain is not valid."""


class ActionNotFoundException(ValueError, RasaException):
    """Raised when an action name could not be found."""


class SessionConfig(NamedTuple):
    session_expiration_time: float  # in minutes
    carry_over_slots: bool

    @staticmethod
    def default() -> "SessionConfig":
        return SessionConfig(
            rasa.shared.constants.DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES,
            rasa.shared.constants.DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION,
        )

    def are_sessions_enabled(self) -> bool:
        return self.session_expiration_time > 0


class Domain:
    """The domain specifies the universe in which the bot's policy acts.

    A Domain subclass provides the actions the bot can take, the intents
    and entities it can recognise."""

    @classmethod
    def empty(cls) -> "Domain":
        return cls([], [], [], {}, [], {})

    @classmethod
    def load(cls, paths: Union[List[Union[Path, Text]], Text, Path]) -> "Domain":
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
    def from_path(cls, path: Union[Text, Path]) -> "Domain":
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

    @classmethod
    def from_yaml(cls, yaml: Text, original_filename: Text = "") -> "Domain":
        try:
            rasa.shared.utils.validation.validate_yaml_schema(
                yaml, rasa.shared.constants.DOMAIN_SCHEMA_FILE
            )

            data = rasa.shared.utils.io.read_yaml(yaml)
            if not rasa.shared.utils.validation.validate_training_data_format_version(
                data, original_filename
            ):
                return Domain.empty()

            return cls.from_dict(data)
        except YamlException as e:
            e.filename = original_filename
            raise e

    @classmethod
    def from_dict(cls, data: Dict) -> "Domain":
        utter_templates = data.get(KEY_RESPONSES, {})
        slots = cls.collect_slots(data.get(KEY_SLOTS, {}))
        additional_arguments = data.get("config", {})
        session_config = cls._get_session_config(data.get(SESSION_CONFIG_KEY, {}))
        intents = data.get(KEY_INTENTS, {})
        forms = data.get(KEY_FORMS, {})

        _validate_slot_mappings(forms)

        return cls(
            intents,
            data.get(KEY_ENTITIES, {}),
            slots,
            utter_templates,
            data.get(KEY_ACTIONS, []),
            data.get(KEY_FORMS, {}),
            data.get(KEY_E2E_ACTIONS, []),
            session_config=session_config,
            **additional_arguments,
        )

    @staticmethod
    def _get_session_config(session_config: Dict) -> SessionConfig:
        session_expiration_time_min = session_config.get(SESSION_EXPIRATION_TIME_KEY)

        if session_expiration_time_min is None:
            session_expiration_time_min = (
                rasa.shared.constants.DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES
            )

        carry_over_slots = session_config.get(
            CARRY_OVER_SLOTS_KEY,
            rasa.shared.constants.DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION,
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
            for key, val in config.items():
                combined["config"][key] = val

        if override or self.session_config == SessionConfig.default():
            combined[SESSION_CONFIG_KEY] = domain_dict[SESSION_CONFIG_KEY]

        combined[KEY_INTENTS] = merge_lists_of_dicts(
            combined[KEY_INTENTS], domain_dict[KEY_INTENTS], override
        )

        # remove existing forms from new actions
        for form in combined[KEY_FORMS]:
            if form in domain_dict[KEY_ACTIONS]:
                domain_dict[KEY_ACTIONS].remove(form)

        for key in [KEY_ENTITIES, KEY_ACTIONS, KEY_E2E_ACTIONS]:
            combined[key] = merge_lists(combined[key], domain_dict[key])

        for key in [KEY_FORMS, KEY_RESPONSES, KEY_SLOTS]:
            combined[key] = merge_dicts(combined[key], domain_dict[key], override)

        return self.__class__.from_dict(combined)

    @staticmethod
    def collect_slots(slot_dict: Dict[Text, Any]) -> List[Slot]:
        slots = []
        # make a copy to not alter the input dictionary
        slot_dict = copy.deepcopy(slot_dict)
        # Sort the slots so that the order of the slot states is consistent
        for slot_name in sorted(slot_dict):
            slot_type = slot_dict[slot_name].pop("type", None)
            slot_class = Slot.resolve_by_type(slot_type)

            slot = slot_class(slot_name, **slot_dict[slot_name])
            slots.append(slot)
        return slots

    @staticmethod
    def _transform_intent_properties_for_internal_use(
        intent: Dict[Text, Any],
        entities: List[Text],
        roles: Dict[Text, List[Text]],
        groups: Dict[Text, List[Text]],
    ) -> Dict[Text, Any]:
        """Transform intent properties coming from a domain file for internal use.

        In domain files, `use_entities` or `ignore_entities` is used. Internally, there
        is a property `used_entities` instead that lists all entities to be used.

        Args:
            intent: The intents as provided by a domain file.
            entities: All entities as provided by a domain file.
            roles: All roles for entities as provided by a domain file.
            groups: All groups for entities as provided by a domain file.

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
        # if the listed entities have a role or group label, concatenate the entity
        # label with the corresponding role or group label to make sure roles and
        # groups can also influence the dialogue predictions
        if properties[USE_ENTITIES_KEY] is True:
            included_entities = set(entities)
            included_entities.update(Domain.concatenate_entity_labels(roles))
            included_entities.update(Domain.concatenate_entity_labels(groups))
        else:
            included_entities = set(properties[USE_ENTITIES_KEY])
            for entity in list(included_entities):
                included_entities.update(
                    Domain.concatenate_entity_labels(roles, entity)
                )
                included_entities.update(
                    Domain.concatenate_entity_labels(groups, entity)
                )
        excluded_entities = set(properties[IGNORE_ENTITIES_KEY])
        for entity in list(excluded_entities):
            excluded_entities.update(Domain.concatenate_entity_labels(roles, entity))
            excluded_entities.update(Domain.concatenate_entity_labels(groups, entity))
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
                docs=f"{rasa.shared.constants.DOCS_URL_DOMAINS}",
            )

        properties[USED_ENTITIES_KEY] = used_entities
        del properties[USE_ENTITIES_KEY]
        del properties[IGNORE_ENTITIES_KEY]

        return intent

    @rasa.shared.utils.common.lazy_property
    def retrieval_intents(self) -> List[Text]:
        """List retrieval intents present in the domain."""
        return [
            intent
            for intent in self.intent_properties
            if self.intent_properties[intent].get(IS_RETRIEVAL_INTENT_KEY)
        ]

    @classmethod
    def collect_entity_properties(
        cls, domain_entities: List[Union[Text, Dict[Text, Any]]]
    ) -> Tuple[List[Text], Dict[Text, List[Text]], Dict[Text, List[Text]]]:
        """Get entity properties for a domain from what is provided by a domain file.

        Args:
            domain_entities: The entities as provided by a domain file.

        Returns:
            A list of entity names.
            A dictionary of entity names to roles.
            A dictionary of entity names to groups.
        """
        entities: List[Text] = []
        roles: Dict[Text, List[Text]] = {}
        groups: Dict[Text, List[Text]] = {}

        for entity in domain_entities:
            if isinstance(entity, str):
                entities.append(entity)
            elif isinstance(entity, dict):
                for _entity, sub_labels in entity.items():
                    entities.append(_entity)
                    if ENTITY_ROLES_KEY in sub_labels:
                        roles[_entity] = sub_labels[ENTITY_ROLES_KEY]
                    if ENTITY_GROUPS_KEY in sub_labels:
                        groups[_entity] = sub_labels[ENTITY_GROUPS_KEY]
            else:
                raise InvalidDomain(
                    f"Invalid domain. Entity is invalid, type of entity '{entity}' "
                    f"not supported: '{type(entity).__name__}'"
                )

        return entities, roles, groups

    @classmethod
    def collect_intent_properties(
        cls,
        intents: List[Union[Text, Dict[Text, Any]]],
        entities: List[Text],
        roles: Dict[Text, List[Text]],
        groups: Dict[Text, List[Text]],
    ) -> Dict[Text, Dict[Text, Union[bool, List]]]:
        """Get intent properties for a domain from what is provided by a domain file.

        Args:
            intents: The intents as provided by a domain file.
            entities: All entities as provided by a domain file.
            roles: The roles of entities as provided by a domain file.
            groups: The groups of entities as provided by a domain file.

        Returns:
            The intent properties to be stored in the domain.
        """
        # make a copy to not alter the input argument
        intents = copy.deepcopy(intents)
        intent_properties = {}
        duplicates = set()

        for intent in intents:
            intent_name, properties = cls._intent_properties(
                intent, entities, roles, groups
            )

            if intent_name in intent_properties.keys():
                duplicates.add(intent_name)

            intent_properties.update(properties)

        if duplicates:
            raise InvalidDomain(
                f"Intents are not unique! Found multiple intents with name(s) {sorted(duplicates)}. "
                f"Either rename or remove the duplicate ones."
            )

        cls._add_default_intents(intent_properties, entities, roles, groups)

        return intent_properties

    @classmethod
    def _intent_properties(
        cls,
        intent: Union[Text, Dict[Text, Any]],
        entities: List[Text],
        roles: Dict[Text, List[Text]],
        groups: Dict[Text, List[Text]],
    ) -> Tuple[Text, Dict[Text, Any]]:
        if not isinstance(intent, dict):
            intent_name = intent
            intent = {intent_name: {USE_ENTITIES_KEY: True, IGNORE_ENTITIES_KEY: []}}
        else:
            intent_name = list(intent.keys())[0]

        return (
            intent_name,
            cls._transform_intent_properties_for_internal_use(
                intent, entities, roles, groups
            ),
        )

    @classmethod
    def _add_default_intents(
        cls,
        intent_properties: Dict[Text, Dict[Text, Union[bool, List]]],
        entities: List[Text],
        roles: Optional[Dict[Text, List[Text]]],
        groups: Optional[Dict[Text, List[Text]]],
    ) -> None:
        for intent_name in rasa.shared.core.constants.DEFAULT_INTENTS:
            if intent_name not in intent_properties:
                _, properties = cls._intent_properties(
                    intent_name, entities, roles, groups
                )
                intent_properties.update(properties)

    def __init__(
        self,
        intents: Union[Set[Text], List[Text], List[Dict[Text, Any]]],
        entities: List[Union[Text, Dict[Text, Any]]],
        slots: List[Slot],
        templates: Dict[Text, List[Dict[Text, Any]]],
        action_names: List[Text],
        forms: Union[Dict[Text, Any], List[Text]],
        action_texts: Optional[List[Text]] = None,
        store_entities_as_slots: bool = True,
        session_config: SessionConfig = SessionConfig.default(),
    ) -> None:
        self.entities, self.roles, self.groups = self.collect_entity_properties(
            entities
        )

        self.intent_properties = self.collect_intent_properties(
            intents, self.entities, self.roles, self.groups
        )
        self.overridden_default_intents = self._collect_overridden_default_intents(
            intents
        )

        self.form_names, self.forms, overridden_form_actions = self._initialize_forms(
            forms
        )
        action_names += overridden_form_actions

        self.slots = slots
        self.templates = templates
        self.action_texts = action_texts or []
        self.session_config = session_config

        self._custom_actions = action_names

        # only includes custom actions and utterance actions
        self.user_actions = self._combine_with_templates(action_names, templates)

        # includes all actions (custom, utterance, default actions and forms)
        self.action_names = (
            self._combine_user_with_default_actions(self.user_actions)
            + [
                form_name
                for form_name in self.form_names
                if form_name not in self._custom_actions
            ]
            + self.action_texts
        )

        self.store_entities_as_slots = store_entities_as_slots
        self._check_domain_sanity()

    def __deepcopy__(self, memo: Optional[Dict[int, Any]]) -> "Domain":
        """Enables making a deep copy of the `Domain` using `copy.deepcopy`.

        See https://docs.python.org/3/library/copy.html#copy.deepcopy
        for more implementation.

        Args:
            memo: Optional dictionary of objects already copied during the current
            copying pass.

        Returns:
            A deep copy of the current domain.
        """
        domain_dict = self.as_dict()
        return self.__class__.from_dict(copy.deepcopy(domain_dict, memo))

    @staticmethod
    def _collect_overridden_default_intents(
        intents: Union[Set[Text], List[Text], List[Dict[Text, Any]]]
    ) -> List[Text]:
        """Collects the default intents overridden by the user.

        Args:
            intents: User-provided intents.

        Returns:
            User-defined intents that are default intents.
        """
        intent_names: Set[Text] = {
            list(intent.keys())[0] if isinstance(intent, dict) else intent
            for intent in intents
        }
        return sorted(intent_names & set(rasa.shared.core.constants.DEFAULT_INTENTS))

    @staticmethod
    def _initialize_forms(
        forms: Union[Dict[Text, Any], List[Text]]
    ) -> Tuple[List[Text], Dict[Text, Any], List[Text]]:
        """Retrieves the initial values for the Domain's form fields.

        Args:
            forms: Form names (if forms are a list) or a form dictionary. Forms
                provided in dictionary format have the form names as keys, and either
                empty dictionaries as values, or objects containing
                `SlotMapping`s.

        Returns:
            The form names, a mapping of form names and slot mappings, and custom
            actions.
            Returning custom actions for each forms means that Rasa Open Source should
            not use the default `FormAction` for the forms, but rather a custom action
            for it. This can e.g. be used to run the deprecated Rasa Open Source 1
            `FormAction` which is implemented in the Rasa SDK.
        """
        if isinstance(forms, dict):
            # dict with slot mappings
            return list(forms.keys()), forms, []

        if isinstance(forms, list) and (not forms or isinstance(forms[0], str)):
            # list of form names (Rasa Open Source 1 format)
            rasa.shared.utils.io.raise_warning(
                "The `forms` section in the domain used the old Rasa Open Source 1 "
                "list format to define forms. Rasa Open Source will be configured to "
                "use the deprecated `FormAction` within the Rasa SDK. If you want to "
                "use the new Rasa Open Source 2 `FormAction` adapt your `forms` "
                "section as described in the documentation. Support for the "
                "deprecated `FormAction` in the Rasa SDK will be removed in Rasa Open "
                "Source 3.0.",
                docs=rasa.shared.constants.DOCS_URL_FORMS,
                category=FutureWarning,
            )
            return forms, {form_name: {} for form_name in forms}, forms

        rasa.shared.utils.io.raise_warning(
            f"The `forms` section in the domain needs to contain a dictionary. "
            f"Instead found an object of type '{type(forms)}'.",
            docs=rasa.shared.constants.DOCS_URL_FORMS,
        )

        return [], {}, []

    def __hash__(self) -> int:
        """Returns a unique hash for the domain."""
        return int(self.fingerprint(), 16)

    def fingerprint(self) -> Text:
        """Returns a unique hash for the domain which is stable across python runs.

        Returns:
            fingerprint of the domain
        """
        self_as_dict = self.as_dict()
        self_as_dict[
            KEY_INTENTS
        ] = rasa.shared.utils.common.sort_list_of_dicts_by_first_key(
            self_as_dict[KEY_INTENTS]
        )
        self_as_dict[KEY_ACTIONS] = self.action_names
        return rasa.shared.utils.io.get_dictionary_fingerprint(self_as_dict)

    @rasa.shared.utils.common.lazy_property
    def user_actions_and_forms(self):
        """Returns combination of user actions and forms."""

        return self.user_actions + self.form_names

    @rasa.shared.utils.common.lazy_property
    def num_actions(self):
        """Returns the number of available actions."""

        # noinspection PyTypeChecker
        return len(self.action_names)

    @rasa.shared.utils.common.lazy_property
    def num_states(self):
        """Number of used input states for the action prediction."""

        return len(self.input_states)

    @rasa.shared.utils.common.lazy_property
    def retrieval_intent_templates(self) -> Dict[Text, List[Dict[Text, Any]]]:
        """Return only the templates which are defined for retrieval intents"""

        return dict(
            filter(
                lambda x: self.is_retrieval_intent_template(x), self.templates.items()
            )
        )

    @staticmethod
    def is_retrieval_intent_template(
        template: Tuple[Text, List[Dict[Text, Any]]]
    ) -> bool:
        """Check if the response template is for a retrieval intent.

        These templates have a `/` symbol in their name. Use that to filter them from the rest.
        """
        return rasa.shared.nlu.constants.RESPONSE_IDENTIFIER_DELIMITER in template[0]

    def add_categorical_slot_default_value(self) -> None:
        """Add a default value to all categorical slots.

        All unseen values found for the slot will be mapped to this default value
        for featurization.
        """
        for slot in [s for s in self.slots if isinstance(s, CategoricalSlot)]:
            slot.add_default_value()

    def add_requested_slot(self) -> None:
        """Add a slot called `requested_slot` to the list of slots.

        The value of this slot will hold the name of the slot which the user
        needs to fill in next (either explicitly or implicitly) as part of a form.
        """
        if self.form_names and rasa.shared.core.constants.REQUESTED_SLOT not in [
            s.name for s in self.slots
        ]:
            self.slots.append(
                TextSlot(
                    rasa.shared.core.constants.REQUESTED_SLOT,
                    influence_conversation=False,
                )
            )

    def add_knowledge_base_slots(self) -> None:
        """
        Add slots for the knowledge base action to the list of slots, if the
        default knowledge base action name is present.

        As soon as the knowledge base action is not experimental anymore, we should
        consider creating a new section in the domain file dedicated to knowledge
        base slots.
        """
        if (
            rasa.shared.core.constants.DEFAULT_KNOWLEDGE_BASE_ACTION
            in self.action_names
        ):
            logger.warning(
                "You are using an experiential feature: Action '{}'!".format(
                    rasa.shared.core.constants.DEFAULT_KNOWLEDGE_BASE_ACTION
                )
            )
            slot_names = [s.name for s in self.slots]
            knowledge_base_slots = [
                rasa.shared.core.constants.SLOT_LISTED_ITEMS,
                rasa.shared.core.constants.SLOT_LAST_OBJECT,
                rasa.shared.core.constants.SLOT_LAST_OBJECT_TYPE,
            ]
            for s in knowledge_base_slots:
                if s not in slot_names:
                    self.slots.append(TextSlot(s, influence_conversation=False))

    def index_for_action(self, action_name: Text) -> Optional[int]:
        """Look up which action index corresponds to this action name."""

        try:
            return self.action_names.index(action_name)
        except ValueError:
            self.raise_action_not_found_exception(action_name)

    def raise_action_not_found_exception(self, action_name) -> NoReturn:
        action_names = "\n".join([f"\t - {a}" for a in self.action_names])
        raise ActionNotFoundException(
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
    @rasa.shared.utils.common.lazy_property
    def slot_states(self) -> List[Text]:
        """Returns all available slot state strings."""

        return [
            f"{slot.name}_{feature_index}"
            for slot in self.slots
            for feature_index in range(0, slot.feature_dimensionality())
        ]

    # noinspection PyTypeChecker
    @rasa.shared.utils.common.lazy_property
    def entity_states(self) -> List[Text]:
        """Returns all available entity state strings."""

        entity_states = copy.deepcopy(self.entities)
        entity_states.extend(Domain.concatenate_entity_labels(self.roles))
        entity_states.extend(Domain.concatenate_entity_labels(self.groups))

        return entity_states

    @staticmethod
    def concatenate_entity_labels(
        entity_labels: Dict[Text, List[Text]], entity: Optional[Text] = None
    ) -> List[Text]:
        """Concatenates the given entity labels with their corresponding sub-labels.

        If a specific entity label is given, only this entity label will be
        concatenated with its corresponding sub-labels.

        Args:
            entity_labels: A map of an entity label to its sub-label list.
            entity: If present, only this entity will be considered.

        Returns:
            A list of labels.
        """
        if entity is not None and entity not in entity_labels:
            return []

        if entity:
            return [
                f"{entity}{rasa.shared.core.constants.ENTITY_LABEL_SEPARATOR}{sub_label}"
                for sub_label in entity_labels[entity]
            ]

        return [
            f"{entity_label}{rasa.shared.core.constants.ENTITY_LABEL_SEPARATOR}{entity_sub_label}"
            for entity_label, entity_sub_labels in entity_labels.items()
            for entity_sub_label in entity_sub_labels
        ]

    @rasa.shared.utils.common.lazy_property
    def input_state_map(self) -> Dict[Text, int]:
        """Provide a mapping from state names to indices."""
        return {f: i for i, f in enumerate(self.input_states)}

    @rasa.shared.utils.common.lazy_property
    def input_states(self) -> List[Text]:
        """Returns all available states."""

        return (
            self.intents
            + self.entity_states
            + self.slot_states
            + self.action_names
            + self.form_names
        )

    def _get_featurized_entities(self, latest_message: UserUttered) -> Set[Text]:
        intent_name = latest_message.intent.get(
            rasa.shared.nlu.constants.INTENT_NAME_KEY
        )
        intent_config = self.intent_config(intent_name)
        entities = latest_message.entities

        # If Entity Roles and Groups is used, we also need to make sure the roles and
        # groups get featurized. We concatenate the entity label with the role/group
        # label using a special separator to make sure that the resulting label is
        # unique (as you can have the same role/group label for different entities).
        entity_names = (
            set(entity["entity"] for entity in entities if "entity" in entity.keys())
            | set(
                f"{entity['entity']}"
                f"{rasa.shared.core.constants.ENTITY_LABEL_SEPARATOR}{entity['role']}"
                for entity in entities
                if "entity" in entity.keys() and "role" in entity.keys()
            )
            | set(
                f"{entity['entity']}"
                f"{rasa.shared.core.constants.ENTITY_LABEL_SEPARATOR}{entity['group']}"
                for entity in entities
                if "entity" in entity.keys() and "group" in entity.keys()
            )
        )

        # the USED_ENTITIES_KEY of an intent also contains the entity labels and the
        # concatenated entity labels with their corresponding roles and groups labels
        wanted_entities = set(intent_config.get(USED_ENTITIES_KEY, entity_names))

        return entity_names & wanted_entities

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
        # proceed with values only if the user of a bot have done something
        # at the previous step i.e., when the state is not empty.
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
            sub_state[rasa.shared.nlu.constants.ENTITIES] = entities
        else:
            sub_state.pop(rasa.shared.nlu.constants.ENTITIES, None)

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
        slots = {}
        for slot_name, slot in tracker.slots.items():
            if slot is not None and slot.as_feature():
                if slot.value == rasa.shared.core.constants.SHOULD_NOT_BE_SET:
                    slots[slot_name] = rasa.shared.core.constants.SHOULD_NOT_BE_SET
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
        active_loop = tracker.active_loop.get(rasa.shared.core.constants.LOOP_NAME)
        if active_loop:
            return {rasa.shared.core.constants.LOOP_NAME: active_loop}
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
            rasa.shared.core.constants.USER: self._get_user_sub_state(tracker),
            rasa.shared.core.constants.SLOTS: self._get_slots_sub_state(tracker),
            rasa.shared.core.constants.PREVIOUS_ACTION: self._get_prev_action_sub_state(
                tracker
            ),
            rasa.shared.core.constants.ACTIVE_LOOP: self._get_active_loop_sub_state(
                tracker
            ),
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
                        e.get("value") for e in entities if e.get("entity") == s.name
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
        rasa.shared.utils.io.create_directory_for_file(domain_spec_path)

        metadata = {"states": self.input_states}
        rasa.shared.utils.io.dump_obj_as_json_to_file(domain_spec_path, metadata)

    @classmethod
    def load_specification(cls, path: Text) -> Dict[Text, Any]:
        """Load a domains specification from a dumped model directory."""
        metadata_path = os.path.join(path, "domain.json")

        return json.loads(rasa.shared.utils.io.read_file(metadata_path))

    def compare_with_specification(self, path: Text) -> bool:
        """Compare the domain spec of the current and the loaded domain.

        Throws exception if the loaded domain specification is different
        to the current domain are different.
        """
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

    def _slot_definitions(self) -> Dict[Any, Dict[str, Any]]:
        return {slot.name: slot.persistence_info() for slot in self.slots}

    def as_dict(self) -> Dict[Text, Any]:
        return {
            "config": {"store_entities_as_slots": self.store_entities_as_slots},
            SESSION_CONFIG_KEY: {
                SESSION_EXPIRATION_TIME_KEY: self.session_config.session_expiration_time,
                CARRY_OVER_SLOTS_KEY: self.session_config.carry_over_slots,
            },
            KEY_INTENTS: self._transform_intents_for_file(),
            KEY_ENTITIES: self._transform_entities_for_file(),
            KEY_SLOTS: self._slot_definitions(),
            KEY_RESPONSES: self.templates,
            KEY_ACTIONS: self._custom_actions,  # class names of the actions
            KEY_FORMS: self.forms,
            KEY_E2E_ACTIONS: self.action_texts,
        }

    def _transform_intents_for_file(self) -> List[Union[Text, Dict[Text, Any]]]:
        """Transform intent properties for displaying or writing into a domain file.

        Internally, there is a property `used_entities` that lists all entities to be
        used. In domain files, `use_entities` or `ignore_entities` is used instead to
        list individual entities to ex- or include, because this is easier to read.

        Returns:
            The intent properties as they are used in domain files.
        """
        intent_properties = copy.deepcopy(self.intent_properties)
        intents_for_file = []

        for intent_name, intent_props in intent_properties.items():
            if (
                intent_name in rasa.shared.core.constants.DEFAULT_INTENTS
                and intent_name not in self.overridden_default_intents
            ):
                # Default intents should be not dumped with the domain
                continue
            # `use_entities` and `ignore_entities` in the domain file do not consider
            # the role and group labels remove them from the list to make sure to not
            # put them into the domain file
            use_entities = set(
                entity
                for entity in intent_props[USED_ENTITIES_KEY]
                if rasa.shared.core.constants.ENTITY_LABEL_SEPARATOR not in entity
            )
            ignore_entities = set(self.entities) - use_entities
            if len(use_entities) == len(self.entities):
                intent_props[USE_ENTITIES_KEY] = True
            elif len(use_entities) <= len(self.entities) / 2:
                intent_props[USE_ENTITIES_KEY] = list(use_entities)
            else:
                intent_props[IGNORE_ENTITIES_KEY] = list(ignore_entities)
            intent_props.pop(USED_ENTITIES_KEY)
            intents_for_file.append({intent_name: intent_props})

        return intents_for_file

    def _transform_entities_for_file(self) -> List[Union[Text, Dict[Text, Any]]]:
        """Transform entity properties for displaying or writing to a domain file.

        Returns:
            The entity properties as they are used in domain files.
        """
        entities_for_file = []

        for entity in self.entities:
            if entity in self.roles and entity in self.groups:
                entities_for_file.append(
                    {
                        entity: {
                            ENTITY_GROUPS_KEY: self.groups[entity],
                            ENTITY_ROLES_KEY: self.roles[entity],
                        }
                    }
                )
            elif entity in self.roles:
                entities_for_file.append(
                    {entity: {ENTITY_ROLES_KEY: self.roles[entity]}}
                )
            elif entity in self.groups:
                entities_for_file.append(
                    {entity: {ENTITY_GROUPS_KEY: self.groups[entity]}}
                )
            else:
                entities_for_file.append(entity)

        return entities_for_file

    def cleaned_domain(self) -> Dict[Text, Any]:
        """Fetch cleaned domain to display or write into a file.

        The internal `used_entities` property is replaced by `use_entities` or
        `ignore_entities` and redundant keys are replaced with default values
        to make the domain easier readable.

        Returns:
            A cleaned dictionary version of the domain.
        """
        domain_data = self.as_dict()
        # remove e2e actions from domain before we display it
        domain_data.pop(KEY_E2E_ACTIONS, None)

        for idx, intent_info in enumerate(domain_data[KEY_INTENTS]):
            for name, intent in intent_info.items():
                if intent.get(USE_ENTITIES_KEY) is True:
                    del intent[USE_ENTITIES_KEY]
                if not intent.get(IGNORE_ENTITIES_KEY):
                    intent.pop(IGNORE_ENTITIES_KEY, None)
                if len(intent) == 0:
                    domain_data[KEY_INTENTS][idx] = name

        for slot in domain_data[KEY_SLOTS].values():
            if slot["initial_value"] is None:
                del slot["initial_value"]
            if slot["auto_fill"]:
                del slot["auto_fill"]
            if slot["type"].startswith("rasa.shared.core.slots"):
                slot["type"] = Slot.resolve_by_type(slot["type"]).type_name

        if domain_data["config"]["store_entities_as_slots"]:
            del domain_data["config"]["store_entities_as_slots"]

        # clean empty keys
        return {
            k: v
            for k, v in domain_data.items()
            if v != {} and v != [] and v is not None
        }

    def persist(self, filename: Union[Text, Path]) -> None:
        """Write domain to a file."""
        as_yaml = self.as_yaml(clean_before_dump=False)
        rasa.shared.utils.io.write_text_file(as_yaml, filename)

    def persist_clean(self, filename: Union[Text, Path]) -> None:
        """Write cleaned domain to a file."""
        as_yaml = self.as_yaml(clean_before_dump=True)
        rasa.shared.utils.io.write_text_file(as_yaml, filename)

    def as_yaml(self, clean_before_dump: bool = False) -> Text:
        """Dump the `Domain` object as a YAML string.
        This function preserves the orders of the keys in the domain.

        Args:
            clean_before_dump: When set to `True`, this method returns
                               a version of the domain without internal
                               information. Defaults to `False`.
        Returns:
            A string in YAML format representing the domain.
        """
        # setting the `version` key first so that it appears at the top of YAML files
        # thanks to the `should_preserve_key_order` argument
        # of `dump_obj_as_yaml_to_string`
        domain_data: Dict[Text, Any] = {
            KEY_TRAINING_DATA_FORMAT_VERSION: rasa.shared.constants.LATEST_TRAINING_DATA_FORMAT_VERSION
        }
        if clean_before_dump:
            domain_data.update(self.cleaned_domain())
        else:
            domain_data.update(self.as_dict())

        return rasa.shared.utils.io.dump_obj_as_yaml_to_string(
            domain_data, should_preserve_key_order=True
        )

    def intent_config(self, intent_name: Text) -> Dict[Text, Any]:
        """Return the configuration for an intent."""
        return self.intent_properties.get(intent_name, {})

    @rasa.shared.utils.common.lazy_property
    def intents(self):
        return sorted(self.intent_properties.keys())

    @property
    def _slots_for_domain_warnings(self) -> List[Text]:
        """Fetch names of slots that are used in domain warnings.

        Excludes slots which aren't featurized.
        """

        return [s.name for s in self.slots if s.influence_conversation]

    @property
    def _actions_for_domain_warnings(self) -> List[Text]:
        """Fetch names of actions that are used in domain warnings.

        Includes user and form actions, but excludes those that are default actions.
        """

        return [
            a
            for a in self.user_actions_and_forms
            if a not in rasa.shared.core.constants.DEFAULT_ACTION_NAMES
        ]

    @staticmethod
    def _get_symmetric_difference(
        domain_elements: Union[List[Text], Set[Text]],
        training_data_elements: Optional[Union[List[Text], Set[Text]]],
    ) -> Dict[Text, Set[Text]]:
        """Get symmetric difference between a set of domain elements and a set of
        training data elements.

        Returns a dictionary containing a list of items found in the `domain_elements`
        but not in `training_data_elements` at key `in_domain`, and a list of items
        found in `training_data_elements` but not in `domain_elements` at key
        `in_training_data_set`.
        """

        if training_data_elements is None:
            training_data_elements = set()

        in_domain_diff = set(domain_elements) - set(training_data_elements)
        in_training_data_diff = set(training_data_elements) - set(domain_elements)

        return {"in_domain": in_domain_diff, "in_training_data": in_training_data_diff}

    @staticmethod
    def _combine_with_templates(
        actions: List[Text], templates: Dict[Text, Any]
    ) -> List[Text]:
        """Combines actions with utter actions listed in responses section."""
        unique_template_names = [
            a for a in sorted(list(templates.keys())) if a not in actions
        ]
        return actions + unique_template_names

    @staticmethod
    def _combine_user_with_default_actions(user_actions: List[Text]) -> List[Text]:
        # remove all user actions that overwrite default actions
        # this logic is a bit reversed, you'd think that we should remove
        # the action name from the default action names if the user overwrites
        # the action, but there are some locations in the code where we
        # implicitly assume that e.g. "action_listen" is always at location
        # 0 in this array. to keep it that way, we remove the duplicate
        # action names from the users list instead of the defaults
        unique_user_actions = [
            a
            for a in user_actions
            if a not in rasa.shared.core.constants.DEFAULT_ACTION_NAMES
        ]
        return rasa.shared.core.constants.DEFAULT_ACTION_NAMES + unique_user_actions

    def domain_warnings(
        self,
        intents: Optional[Union[List[Text], Set[Text]]] = None,
        entities: Optional[Union[List[Text], Set[Text]]] = None,
        actions: Optional[Union[List[Text], Set[Text]]] = None,
        slots: Optional[Union[List[Text], Set[Text]]] = None,
    ) -> Dict[Text, Dict[Text, Set[Text]]]:
        """Generate domain warnings from intents, entities, actions and slots.

        Returns a dictionary with entries for `intent_warnings`,
        `entity_warnings`, `action_warnings` and `slot_warnings`. Excludes domain slots
        from domain warnings in case they are not featurized.
        """

        intent_warnings = self._get_symmetric_difference(self.intents, intents)
        entity_warnings = self._get_symmetric_difference(self.entities, entities)
        action_warnings = self._get_symmetric_difference(
            self._actions_for_domain_warnings, actions
        )
        slot_warnings = self._get_symmetric_difference(
            self._slots_for_domain_warnings, slots
        )

        return {
            "intent_warnings": intent_warnings,
            "entity_warnings": entity_warnings,
            "action_warnings": action_warnings,
            "slot_warnings": slot_warnings,
        }

    def _check_domain_sanity(self) -> None:
        """Make sure the domain is properly configured.
        If the domain contains any duplicate slots, intents, actions
        or entities, an InvalidDomain error is raised.  This error
        is also raised when intent-action mappings are incorrectly
        named or an utterance template is missing."""

        def get_duplicates(my_items):
            """Returns a list of duplicate items in my_items."""

            return [
                item
                for item, count in collections.Counter(my_items).items()
                if count > 1
            ]

        def check_mappings(
            intent_properties: Dict[Text, Dict[Text, Union[bool, List]]]
        ) -> List[Tuple[Text, Text]]:
            """Check whether intent-action mappings use proper action names."""

            incorrect = []
            for intent, properties in intent_properties.items():
                if "triggers" in properties:
                    triggered_action = properties.get("triggers")
                    if triggered_action not in self.action_names:
                        incorrect.append((intent, str(triggered_action)))
            return incorrect

        def get_exception_message(
            duplicates: Optional[List[Tuple[List[Text], Text]]] = None,
            mappings: List[Tuple[Text, Text]] = None,
        ):
            """Return a message given a list of error locations."""

            message = ""
            if duplicates:
                message += get_duplicate_exception_message(duplicates)
            if mappings:
                if message:
                    message += "\n"
                message += get_mapping_exception_message(mappings)
            return message

        def get_mapping_exception_message(mappings: List[Tuple[Text, Text]]):
            """Return a message given a list of duplicates."""

            message = ""
            for name, action_name in mappings:
                if message:
                    message += "\n"
                message += (
                    "Intent '{}' is set to trigger action '{}', which is "
                    "not defined in the domain.".format(name, action_name)
                )
            return message

        def get_duplicate_exception_message(
            duplicates: List[Tuple[List[Text], Text]]
        ) -> Text:
            """Return a message given a list of duplicates."""

            message = ""
            for d, name in duplicates:
                if d:
                    if message:
                        message += "\n"
                    message += (
                        f"Duplicate {name} in domain. "
                        f"These {name} occur more than once in "
                        f"the domain: '{', '.join(d)}'."
                    )
            return message

        duplicate_actions = get_duplicates(self.action_names)
        duplicate_slots = get_duplicates([s.name for s in self.slots])
        duplicate_entities = get_duplicates(self.entities)
        incorrect_mappings = check_mappings(self.intent_properties)

        if (
            duplicate_actions
            or duplicate_slots
            or duplicate_entities
            or incorrect_mappings
        ):
            raise InvalidDomain(
                get_exception_message(
                    [
                        (duplicate_actions, KEY_ACTIONS),
                        (duplicate_slots, KEY_SLOTS),
                        (duplicate_entities, KEY_ENTITIES),
                    ],
                    incorrect_mappings,
                )
            )

    def check_missing_templates(self) -> None:
        """Warn user of utterance names which have no specified template."""

        utterances = [
            a
            for a in self.action_names
            if a.startswith(rasa.shared.constants.UTTER_PREFIX)
        ]

        missing_templates = [t for t in utterances if t not in self.templates.keys()]

        if missing_templates:
            for template in missing_templates:
                rasa.shared.utils.io.raise_warning(
                    f"Action '{template}' is listed as a "
                    f"response action in the domain file, but there is "
                    f"no matching response defined. Please "
                    f"check your domain.",
                    docs=rasa.shared.constants.DOCS_URL_RESPONSES,
                )

    def is_empty(self) -> bool:
        """Check whether the domain is empty."""
        return self.as_dict() == Domain.empty().as_dict()

    @staticmethod
    def is_domain_file(filename: Text) -> bool:
        """Checks whether the given file path is a Rasa domain file.

        Args:
            filename: Path of the file which should be checked.

        Returns:
            `True` if it's a domain file, otherwise `False`.

        Raises:
            YamlException: if the file seems to be a YAML file (extension) but
                can not be read / parsed.
        """
        from rasa.shared.data import is_likely_yaml_file

        if not is_likely_yaml_file(filename):
            return False

        content = rasa.shared.utils.io.read_yaml_file(filename)
        return any(key in content for key in ALL_DOMAIN_KEYS)

    def slot_mapping_for_form(self, form_name: Text) -> Dict[Text, Any]:
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
        return self.forms.get(form_name, {})


class SlotMapping(Enum):
    """Defines the available slot mappings."""

    FROM_ENTITY = 0
    FROM_INTENT = 1
    FROM_TRIGGER_INTENT = 2
    FROM_TEXT = 3

    def __str__(self) -> Text:
        """Returns a string representation of the object."""
        return self.name.lower()

    @staticmethod
    def validate(mapping: Dict[Text, Any], form_name: Text, slot_name: Text) -> None:
        """Validates a slot mapping.

        Args:
            mapping: The mapping which is validated.
            form_name: The name of the form which uses this slot mapping.
            slot_name: The name of the slot which is mapped by this mapping.

        Raises:
            InvalidDomain: In case the slot mapping is not valid.
        """
        if not isinstance(mapping, dict):
            raise InvalidDomain(
                f"Please make sure that the slot mappings for slot '{slot_name}' in "
                f"your form '{form_name}' are valid dictionaries. Please see "
                f"{rasa.shared.constants.DOCS_URL_FORMS} for more information."
            )

        validations = {
            str(SlotMapping.FROM_ENTITY): ["entity"],
            str(SlotMapping.FROM_INTENT): ["value"],
            str(SlotMapping.FROM_TRIGGER_INTENT): ["value"],
            str(SlotMapping.FROM_TEXT): [],
        }

        mapping_type = mapping.get("type")
        required_keys = validations.get(mapping_type)

        if required_keys is None:
            raise InvalidDomain(
                f"Your form '{form_name}' uses an invalid slot mapping of type "
                f"'{mapping_type}' for slot '{slot_name}'. Please see "
                f"{rasa.shared.constants.DOCS_URL_FORMS} for more information."
            )

        for required_key in required_keys:
            if mapping.get(required_key) is None:
                raise InvalidDomain(
                    f"You need to specify a value for the key "
                    f"'{required_key}' in the slot mapping of type '{mapping_type}' "
                    f"for slot '{slot_name}' in the form '{form_name}'. Please see "
                    f"{rasa.shared.constants.DOCS_URL_FORMS} for more information."
                )


def _validate_slot_mappings(forms: Union[Dict, List]) -> None:
    if isinstance(forms, list):
        if not all(isinstance(form_name, str) for form_name in forms):
            raise InvalidDomain(
                f"If you use the deprecated list syntax for forms, "
                f"all form names have to be strings. Please see "
                f"{rasa.shared.constants.DOCS_URL_FORMS} for more information."
            )

        return

    if not isinstance(forms, dict):
        raise InvalidDomain("Forms have to be specified as dictionary.")

    for form_name, slots in forms.items():
        if slots is None:
            continue

        if not isinstance(slots, Dict):
            raise InvalidDomain(
                f"The slots for form '{form_name}' were specified "
                f"as '{type(slots)}'. They need to be specified "
                f"as dictionary. Please see {rasa.shared.constants.DOCS_URL_FORMS} "
                f"for more information."
            )

        for slot_name, slot_mappings in slots.items():
            if not isinstance(slot_mappings, list):
                raise InvalidDomain(
                    f"The slot mappings for slot '{slot_name}' in "
                    f"form '{form_name}' have type "
                    f"'{type(slot_mappings)}'. It is required to "
                    f"provide a list of slot mappings. Please see "
                    f"{rasa.shared.constants.DOCS_URL_FORMS} for more information."
                )
            for slot_mapping in slot_mappings:
                SlotMapping.validate(slot_mapping, form_name, slot_name)
