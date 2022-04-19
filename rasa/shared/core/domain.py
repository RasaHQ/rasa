import copy
import collections
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    NoReturn,
    Optional,
    Set,
    Text,
    Tuple,
    Union,
    TYPE_CHECKING,
    Iterable,
)

from rasa.shared.constants import (
    DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES,
    DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION,
    DOMAIN_SCHEMA_FILE,
    DOCS_URL_DOMAINS,
    DOCS_URL_FORMS,
    DOCS_URL_MIGRATION_GUIDE,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
    DOCS_URL_RESPONSES,
    REQUIRED_SLOTS_KEY,
    IGNORED_INTENTS,
)
import rasa.shared.core.constants
from rasa.shared.exceptions import RasaException, YamlException, YamlSyntaxException
import rasa.shared.utils.validation
import rasa.shared.utils.io
import rasa.shared.utils.common
from rasa.shared.core.events import SlotSet, UserUttered
from rasa.shared.core.slots import Slot, CategoricalSlot, TextSlot, AnySlot
from rasa.shared.utils.validation import KEY_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.constants import RESPONSE_CONDITION
from rasa.shared.nlu.constants import (
    RESPONSE_IDENTIFIER_DELIMITER,
    INTENT_NAME_KEY,
    ENTITIES,
)

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
KEY_RESPONSES_TEXT = "text"

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
    """The Session Configuration."""

    session_expiration_time: float  # in minutes
    carry_over_slots: bool

    @staticmethod
    def default() -> "SessionConfig":
        """Returns the SessionConfig with the default values."""
        return SessionConfig(
            DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES,
            DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION,
        )

    def are_sessions_enabled(self) -> bool:
        """Returns a boolean value depending on the value of session_expiration_time."""
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
        """Loads the `Domain` from a YAML file."""
        return cls.from_yaml(rasa.shared.utils.io.read_file(path), path)

    @classmethod
    def from_yaml(cls, yaml: Text, original_filename: Text = "") -> "Domain":
        """Loads the `Domain` from YAML text after validating it."""
        try:
            rasa.shared.utils.validation.validate_yaml_schema(yaml, DOMAIN_SCHEMA_FILE)

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
        """Deserializes and creates domain.

        Args:
            data: The serialized domain.

        Returns:
            The instantiated `Domain` object.
        """
        responses = data.get(KEY_RESPONSES, {})
        slots = cls.collect_slots(data.get(KEY_SLOTS, {}))
        additional_arguments = data.get("config", {})
        session_config = cls._get_session_config(data.get(SESSION_CONFIG_KEY, {}))
        intents = data.get(KEY_INTENTS, {})
        forms = data.get(KEY_FORMS, {})
        duplicates = data.get("duplicates", None)

        _validate_slot_mappings(forms)

        return cls(
            intents,
            data.get(KEY_ENTITIES, {}),
            slots,
            responses,
            data.get(KEY_ACTIONS, []),
            data.get(KEY_FORMS, {}),
            data.get(KEY_E2E_ACTIONS, []),
            session_config=session_config,
            duplicates=duplicates,
            **additional_arguments,
        )

    @staticmethod
    def _get_session_config(session_config: Dict) -> SessionConfig:
        session_expiration_time_min = session_config.get(SESSION_EXPIRATION_TIME_KEY)

        if session_expiration_time_min is None:
            session_expiration_time_min = DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES

        carry_over_slots = session_config.get(
            CARRY_OVER_SLOTS_KEY, DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION,
        )

        return SessionConfig(session_expiration_time_min, carry_over_slots)

    @classmethod
    def from_directory(cls, path: Text) -> "Domain":
        """Loads and merges multiple domain files recursively from a directory tree."""
        domain_dict = {}
        for root, _, files in os.walk(path, followlinks=True):
            for file in files:
                full_path = os.path.join(root, file)
                if Domain.is_domain_file(full_path):
                    _ = Domain.from_file(full_path)  # does the validation here only
                    other_dict = rasa.shared.utils.io.read_yaml(
                        rasa.shared.utils.io.read_file(full_path)
                    )
                    domain_dict = Domain.merge_domain_dicts(
                        cls, domain_dict, other_dict
                    )
        domain = Domain.from_dict(domain_dict)
        return domain

    def merge(self, domain: Optional["Domain"], override: bool = False) -> "Domain":
        """Merge this domain with another one, combining their attributes.

        List attributes like ``intents`` and ``actions`` will be deduped
        and merged. Single attributes will be taken from `self` unless
        override is `True`, in which case they are taken from `domain`.
        """
        if not domain or domain.is_empty():
            return self

        if self.is_empty():
            return domain

        domain_dict = domain.as_dict()
        combined = self.as_dict()

        if override:
            config = domain_dict["config"]
            for key, val in config.items():
                combined["config"][key] = val

        if override or self.session_config == SessionConfig.default():
            combined[SESSION_CONFIG_KEY] = domain_dict[SESSION_CONFIG_KEY]

        for key in [KEY_INTENTS, KEY_ENTITIES]:
            if combined[key] or domain_dict[key]:
                combined[key] = self.merge_lists_of_dicts(
                    combined[key], domain_dict[key], override
                )

        # remove existing forms from new actions
        for form in combined[KEY_FORMS]:
            if form in domain_dict[KEY_ACTIONS]:
                domain_dict[KEY_ACTIONS].remove(form)

        for key in [KEY_ACTIONS, KEY_E2E_ACTIONS]:
            combined[key] = self.merge_lists(combined[key], domain_dict[key])

        for key in [KEY_FORMS, KEY_RESPONSES, KEY_SLOTS]:
            combined[key] = self.merge_dicts(combined[key], domain_dict[key], override)

        return self.__class__.from_dict(combined)

    def merge_domain_dicts(
        self, domain1: Dict, domain2: Dict, override: bool = False
    ) -> Dict[Text, Any]:
        """Merges this domain dict with another one, combining their attributes.

        This is used when multiple domain yml files are configured in a single
        directory. Unlike the merge method above, which merges Domain objects by
        creating each object then merging it with the previous, this method merges
        domain dicts, and ensures all attributes (like ``intents``, ``entities``, and
        ``actions``) are known to the Domain when the object is created.

        List attributes like ``intents`` and ``actions`` are deduped
        and merged. Single attributes are taken from `domain1` unless
        override is `True`, in which case they are taken from `domain2`.
        """
        if not domain2:
            return domain1

        if not domain1:
            return domain2

        domain_dict = domain2
        combined = domain1

        if override:
            config = domain_dict["config"]
            for key, val in config.items():
                combined["config"][key] = val

        if override or domain2.get("session_config"):
            combined[SESSION_CONFIG_KEY] = domain_dict[SESSION_CONFIG_KEY]

        duplicates: Dict[Text, List[Text]] = {}

        for key in [KEY_INTENTS, KEY_ENTITIES]:
            if combined.get(key) or domain_dict.get(key):
                duplicates[key] = self.extract_duplicates(
                    combined.get(key, []), domain_dict.get(key, [])
                )
                combined[key] = combined.get(key, [])
                domain_dict[key] = domain_dict.get(key, [])
                combined[key] = self.merge_lists_of_dicts(
                    combined[key], domain_dict[key], override
                )

        # remove existing forms from new actions
        for form in combined.get(KEY_FORMS, []):
            if form in domain_dict.get(KEY_ACTIONS, []):
                domain_dict[KEY_ACTIONS].remove(form)

        for key in [KEY_ACTIONS, KEY_E2E_ACTIONS]:
            duplicates[key] = self.extract_duplicates(
                combined.get(key, []), domain_dict.get(key, [])
            )
            combined[key] = self.merge_lists(
                combined.get(key, []), domain_dict.get(key, [])
            )

        for key in [KEY_FORMS, KEY_RESPONSES, KEY_SLOTS]:
            duplicates[key] = self.extract_duplicates(
                combined.get(key, []), domain_dict.get(key, [])
            )
            combined[key] = self.merge_dicts(
                combined.get(key, {}), domain_dict.get(key, {}), override
            )

        if duplicates:
            duplicates = self.clean_duplicates(duplicates)
            combined.update({"duplicates": duplicates})

        return combined

    @staticmethod
    def extract_duplicates(list1: List[Any], list2: List[Any]) -> List[Any]:
        """Extracts duplicates from two lists."""
        if list1:
            dict1 = {
                (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i
                for i in list1
            }
        else:
            dict1 = {}

        if list2:
            dict2 = {
                (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i
                for i in list2
            }
        else:
            dict2 = {}

        set1 = set(dict1.keys())
        set2 = set(dict2.keys())
        dupes = set1.intersection(set2)
        return sorted(list(dupes))

    @staticmethod
    def clean_duplicates(dupes: Dict[Text, Any]) -> Dict[Text, Any]:
        """Removes keys for empty values."""
        duplicates = dupes.copy()
        for k in dupes:
            if not dupes[k]:
                duplicates.pop(k)

        return duplicates

    @staticmethod
    def merge_dicts(
        tempDict1: Dict[Text, Any],
        tempDict2: Dict[Text, Any],
        override_existing_values: bool = False,
    ) -> Dict[Text, Any]:
        """Merges two dicts."""
        if override_existing_values:
            merged_dicts, b = tempDict1.copy(), tempDict2.copy()

        else:
            merged_dicts, b = tempDict2.copy(), tempDict1.copy()
        merged_dicts.update(b)
        return merged_dicts

    @staticmethod
    def merge_lists(list1: List[Any], list2: List[Any]) -> List[Any]:
        """Merges 2 lists."""
        return sorted(list(set(list1 + list2)))

    @staticmethod
    def merge_lists_of_dicts(
        dict_list1: List[Dict],
        dict_list2: List[Dict],
        override_existing_values: bool = False,
    ) -> List[Dict]:
        """Merges 2 dict lists."""
        dict1 = {
            (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i
            for i in dict_list1
        }
        dict2 = {
            (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i
            for i in dict_list2
        }
        merged_dicts = Domain.merge_dicts(dict1, dict2, override_existing_values)
        return list(merged_dicts.values())

    @staticmethod
    def collect_slots(slot_dict: Dict[Text, Any]) -> List[Slot]:
        """Collects the slots."""
        slots = []
        # make a copy to not alter the input dictionary
        slot_dict = copy.deepcopy(slot_dict)
        # Don't sort the slots, see https://github.com/RasaHQ/rasa-x/issues/3900
        for slot_name in slot_dict:
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
        """Transforms the intent's parameters in a format suitable for internal use.

        When an intent is retrieved from the `domain.yml` file, it contains two
        parameters, the `use_entities` and the `ignore_entities` parameter. With
        the values of these two parameters the Domain class is updated, a new
        parameter is added to the intent called `used_entities` and the two
        previous parameters are deleted. This happens because internally only the
        parameter `used_entities` is needed to list all the entities that should be
        used for this intent.

        Args:
            intent: The intent as retrieved from the `domain.yml` file thus having two
                parameters, the `use_entities` and the `ignore_entities` parameter.
            entities: All entities as provided by a domain file.
            roles: All roles for entities as provided by a domain file.
            groups: All groups for entities as provided by a domain file.

        Returns:
            The intent with the new format thus having only one parameter called
            `used_entities` since this is the expected format of the intent
            when used internally.
        """
        name, properties = list(intent.items())[0]

        if properties:
            properties.setdefault(USE_ENTITIES_KEY, True)
        else:
            raise InvalidDomain(
                f"In the `domain.yml` file, the intent '{name}' cannot have value of"
                f" `{type(properties)}`. If you have placed a ':' character after the"
                f" intent's name without adding any additional parameters to this"
                f" intent then you would need to remove the ':' character. Please see"
                f" {rasa.shared.constants.DOCS_URL_DOMAINS} for more information on how"
                f" to correctly add `intents` in the `domain` and"
                f" {rasa.shared.constants.DOCS_URL_INTENTS} for examples on"
                f" when to use the ':' character after an intent's name."
            )

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
                docs=f"{DOCS_URL_DOMAINS}",
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
                    if sub_labels:
                        if ENTITY_ROLES_KEY in sub_labels:
                            roles[_entity] = sub_labels[ENTITY_ROLES_KEY]
                        if ENTITY_GROUPS_KEY in sub_labels:
                            groups[_entity] = sub_labels[ENTITY_GROUPS_KEY]
                    else:
                        raise InvalidDomain(
                            f"In the `domain.yml` file, the entity '{_entity}' cannot"
                            f" have value of `{type(sub_labels)}`. If you have placed a"
                            f" ':' character after the entity `{_entity}` without"
                            f" adding any additional parameters to this entity then you"
                            f" would need to remove the ':' character. Please see"
                            f" {rasa.shared.constants.DOCS_URL_DOMAINS} for more"
                            f" information on how to correctly add `entities` in the"
                            f" `domain` and {rasa.shared.constants.DOCS_URL_ENTITIES}"
                            f" for examples on when to use the ':' character after an"
                            f" entity's name."
                        )
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
                f"Intents are not unique! Found multiple intents "
                f"with name(s) {sorted(duplicates)}. "
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
        responses: Dict[Text, List[Dict[Text, Any]]],
        action_names: List[Text],
        forms: Union[Dict[Text, Any], List[Text]],
        action_texts: Optional[List[Text]] = None,
        store_entities_as_slots: bool = True,
        session_config: SessionConfig = SessionConfig.default(),
        duplicates: Optional[Dict[Text, List[Text]]] = None,
    ) -> None:
        """Creates a `Domain`.

        Args:
            intents: Intent labels.
            entities: The names of entities which might be present in user messages.
            slots: Slots to store information during the conversation.
            responses: Bot responses. If an action with the same name is executed, it
                will send the matching response to the user.
            action_names: Names of custom actions.
            forms: Form names and their slot mappings.
            action_texts: End-to-End bot utterances from end-to-end stories.
            store_entities_as_slots: If `True` Rasa will automatically create `SlotSet`
                events for entities if there are slots with the same name as the entity.
            session_config: Configuration for conversation sessions. Conversations are
                restarted at the end of a session.
            duplicates: A dictionary where keys are `intents`, `slots`, `forms` and
                `responses` and values are lists of duplicated entries of a
                corresponding type when the domain is built from multiple files.
        """
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

        self.responses = responses

        self.action_texts = action_texts or []
        self.session_config = session_config
        self.duplicates = duplicates

        self._custom_actions = action_names

        # only includes custom actions and utterance actions
        self.user_actions = self._combine_with_responses(action_names, responses)

        # includes all action names (custom, utterance, default actions and forms)
        # and action texts from end-to-end bot utterances
        self.action_names_or_texts = (
            self._combine_user_with_default_actions(self.user_actions)
            + [
                form_name
                for form_name in self.form_names
                if form_name not in self._custom_actions
            ]
            + self.action_texts
        )

        self._user_slots = copy.copy(slots)
        self.slots = slots
        self._add_default_slots()
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

    def count_conditional_response_variations(self) -> int:
        """Returns count of conditional response variations."""
        count = 0
        for response_variations in self.responses.values():
            for variation in response_variations:
                if RESPONSE_CONDITION in variation:
                    count += 1

        return count

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
        return sorted(
            intent_names.intersection(set(rasa.shared.core.constants.DEFAULT_INTENTS))
        )

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
            for form_name, form_data in forms.items():
                if form_data is not None and REQUIRED_SLOTS_KEY not in form_data:
                    forms[form_name] = {REQUIRED_SLOTS_KEY: form_data}
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
            docs=DOCS_URL_FORMS,
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
        self_as_dict[KEY_ACTIONS] = self.action_names_or_texts
        return rasa.shared.utils.io.get_dictionary_fingerprint(self_as_dict)

    @rasa.shared.utils.common.lazy_property
    def user_actions_and_forms(self) -> List[Text]:
        """Returns combination of user actions and forms."""
        return self.user_actions + self.form_names

    @rasa.shared.utils.common.lazy_property
    def action_names(self) -> List[Text]:
        """Returns action names or texts."""
        # Raise `DeprecationWarning` instead of `FutureWarning` as we only want to
        # notify developers about the deprecation (e.g. developers who are using the
        # Python API or writing custom policies). End users can't change anything
        # about this warning except making their developers change any custom code
        # which calls this.
        rasa.shared.utils.io.raise_warning(
            f"{Domain.__name__}.{Domain.action_names.__name__} "
            f"is deprecated and will be removed version 3.0.0.",
            category=DeprecationWarning,
        )
        return self.action_names_or_texts

    @rasa.shared.utils.common.lazy_property
    def num_actions(self) -> int:
        """Returns the number of available actions."""
        # noinspection PyTypeChecker
        return len(self.action_names_or_texts)

    @rasa.shared.utils.common.lazy_property
    def num_states(self) -> int:
        """Number of used input states for the action prediction."""
        return len(self.input_states)

    @rasa.shared.utils.common.lazy_property
    def retrieval_intent_templates(self) -> Dict[Text, List[Dict[Text, Any]]]:
        """Return only the responses which are defined for retrieval intents."""
        rasa.shared.utils.io.raise_deprecation_warning(
            "The terminology 'template' is deprecated and replaced by "
            "'response', call `retrieval_intent_responses` instead of "
            "`retrieval_intent_templates`.",
            docs=f"{DOCS_URL_MIGRATION_GUIDE}#rasa-23-to-rasa-24",
        )
        return self.retrieval_intent_responses

    @rasa.shared.utils.common.lazy_property
    def retrieval_intent_responses(self) -> Dict[Text, List[Dict[Text, Any]]]:
        """Return only the responses which are defined for retrieval intents."""
        return dict(
            filter(
                lambda intent_response: self.is_retrieval_intent_response(
                    intent_response
                ),
                self.responses.items(),
            )
        )

    @rasa.shared.utils.common.lazy_property
    def templates(self) -> Dict[Text, List[Dict[Text, Any]]]:
        """Temporary property before templates become completely deprecated."""
        rasa.shared.utils.io.raise_deprecation_warning(
            "The terminology 'template' is deprecated and replaced by 'response'. "
            "Instead of using the `templates` property, "
            "please use the `responses` property instead.",
            docs=f"{DOCS_URL_MIGRATION_GUIDE}#rasa-23-to-rasa-24",
        )
        return self.responses

    @staticmethod
    def is_retrieval_intent_template(
        response: Tuple[Text, List[Dict[Text, Any]]]
    ) -> bool:
        """Check if the response is for a retrieval intent.

        These templates have a `/` symbol in their name. Use that to filter them from
        the rest.
        """
        rasa.shared.utils.io.raise_deprecation_warning(
            "The terminology 'template' is deprecated "
            "and replaced by 'response', "
            "call `is_retrieval_intent_response` "
            "instead of `is_retrieval_intent_template`.",
            docs=f"{DOCS_URL_MIGRATION_GUIDE}#rasa-23-to-rasa-24",
        )
        return rasa.shared.nlu.constants.RESPONSE_IDENTIFIER_DELIMITER in response[0]

    @staticmethod
    def is_retrieval_intent_response(
        response: Tuple[Text, List[Dict[Text, Any]]]
    ) -> bool:
        """Check if the response is for a retrieval intent.

        These responses have a `/` symbol in their name. Use that to filter them from
        the rest.
        """
        return RESPONSE_IDENTIFIER_DELIMITER in response[0]

    def _add_default_slots(self) -> None:
        """Sets up the default slots and slot values for the domain."""
        self._add_requested_slot()
        self._add_knowledge_base_slots()
        self._add_categorical_slot_default_value()
        self._add_session_metadata_slot()

    def _add_categorical_slot_default_value(self) -> None:
        """Add a default value to all categorical slots.

        All unseen values found for the slot will be mapped to this default value
        for featurization.
        """
        for slot in [s for s in self.slots if isinstance(s, CategoricalSlot)]:
            slot.add_default_value()

    def add_categorical_slot_default_value(self) -> None:
        """See `_add_categorical_slot_default_value` for docstring."""
        rasa.shared.utils.io.raise_deprecation_warning(
            f"'{self.add_categorical_slot_default_value.__name__}' is deprecated and "
            f"will be removed in Rasa Open Source 3.0.0. This method is now "
            f"automatically called when the Domain is created which makes a manual "
            f"call superfluous."
        )
        self._add_categorical_slot_default_value()

    def _add_requested_slot(self) -> None:
        """Add a slot called `requested_slot` to the list of slots.

        The value of this slot will hold the name of the slot which the user
        needs to fill in next (either explicitly or implicitly) as part of a form.
        """
        if self.form_names and rasa.shared.core.constants.REQUESTED_SLOT not in [
            slot.name for slot in self.slots
        ]:
            self.slots.append(
                TextSlot(
                    rasa.shared.core.constants.REQUESTED_SLOT,
                    influence_conversation=False,
                )
            )

    def add_requested_slot(self) -> None:
        """See `_add_categorical_slot_default_value` for docstring."""
        rasa.shared.utils.io.raise_deprecation_warning(
            f"'{self.add_requested_slot.__name__}' is deprecated and "
            f"will be removed in Rasa Open Source 3.0.0. This method is now "
            f"automatically called when the Domain is created which makes a manual "
            f"call superfluous."
        )
        self._add_requested_slot()

    def _add_knowledge_base_slots(self) -> None:
        """Add slots for the knowledge base action to slots.

        Slots are only added if the default knowledge base action name is present.

        As soon as the knowledge base action is not experimental anymore, we should
        consider creating a new section in the domain file dedicated to knowledge
        base slots.
        """
        if (
            rasa.shared.core.constants.DEFAULT_KNOWLEDGE_BASE_ACTION
            in self.action_names_or_texts
        ):
            logger.warning(
                "You are using an experiential feature: Action '{}'!".format(
                    rasa.shared.core.constants.DEFAULT_KNOWLEDGE_BASE_ACTION
                )
            )
            slot_names = [slot.name for slot in self.slots]
            knowledge_base_slots = [
                rasa.shared.core.constants.SLOT_LISTED_ITEMS,
                rasa.shared.core.constants.SLOT_LAST_OBJECT,
                rasa.shared.core.constants.SLOT_LAST_OBJECT_TYPE,
            ]
            for slot in knowledge_base_slots:
                if slot not in slot_names:
                    self.slots.append(TextSlot(slot, influence_conversation=False))

    def add_knowledge_base_slots(self) -> None:
        """See `_add_categorical_slot_default_value` for docstring."""
        rasa.shared.utils.io.raise_deprecation_warning(
            f"'{self.add_knowledge_base_slots.__name__}' is deprecated and "
            f"will be removed in Rasa Open Source 3.0.0. This method is now "
            f"automatically called when the Domain is created which makes a manual "
            f"call superfluous."
        )
        self._add_knowledge_base_slots()

    def _add_session_metadata_slot(self) -> None:
        self.slots.append(
            AnySlot(rasa.shared.core.constants.SESSION_START_METADATA_SLOT,)
        )

    def index_for_action(self, action_name: Text) -> int:
        """Looks up which action index corresponds to this action name."""
        try:
            return self.action_names_or_texts.index(action_name)
        except ValueError:
            self.raise_action_not_found_exception(action_name)

    def raise_action_not_found_exception(self, action_name_or_text: Text) -> NoReturn:
        """Raises exception if action name or text not part of the domain or stories.

        Args:
            action_name_or_text: Name of an action or its text in case it's an
                end-to-end bot utterance.

        Raises:
            ActionNotFoundException: If `action_name_or_text` are not part of this
                domain.
        """
        action_names = "\n".join([f"\t - {a}" for a in self.action_names_or_texts])
        raise ActionNotFoundException(
            f"Cannot access action '{action_name_or_text}', "
            f"as that name is not a registered "
            f"action for this domain. "
            f"Available actions are: \n{action_names}"
        )

    def random_template_for(self, utter_action: Text) -> Optional[Dict[Text, Any]]:
        """Returns a random response for an action name.

        Args:
            utter_action: The name of the utter action.

        Returns:
            A response for an utter action.
        """
        import numpy as np

        # Raise `DeprecationWarning` instead of `FutureWarning` as we only want to
        # notify developers about the deprecation (e.g. developers who are using the
        # Python API or writing custom policies). End users can't change anything
        # about this warning except making their developers change any custom code
        # which calls this.
        rasa.shared.utils.io.raise_warning(
            f"'{Domain.__name__}.{Domain.random_template_for.__class__}' "
            f"is deprecated and will be removed version 3.0.0.",
            category=DeprecationWarning,
        )
        if utter_action in self.responses:
            return np.random.choice(self.responses[utter_action])
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
                f"{entity}"
                f"{rasa.shared.core.constants.ENTITY_LABEL_SEPARATOR}"
                f"{sub_label}"
                for sub_label in entity_labels[entity]
            ]

        return [
            f"{entity_label}"
            f"{rasa.shared.core.constants.ENTITY_LABEL_SEPARATOR}"
            f"{entity_sub_label}"
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
            + self.action_names_or_texts
            + self.form_names
        )

    def _get_featurized_entities(self, latest_message: UserUttered) -> Set[Text]:
        """Gets the names of all entities that are present and wanted in the message.

        Wherever an entity has a role or group specified as well, an additional role-
        or group-specific entity name is added.
        """
        intent_name = latest_message.intent.get(INTENT_NAME_KEY)
        intent_config = self.intent_config(intent_name)
        entities = latest_message.entities

        # If Entity Roles and Groups is used, we also need to make sure the roles and
        # groups get featurized. We concatenate the entity label with the role/group
        # label using a special separator to make sure that the resulting label is
        # unique (as you can have the same role/group label for different entities).
        entity_names_basic = set(
            entity["entity"] for entity in entities if "entity" in entity.keys()
        )
        entity_names_roles = set(
            f"{entity['entity']}"
            f"{rasa.shared.core.constants.ENTITY_LABEL_SEPARATOR}{entity['role']}"
            for entity in entities
            if "entity" in entity.keys() and "role" in entity.keys()
        )
        entity_names_groups = set(
            f"{entity['entity']}"
            f"{rasa.shared.core.constants.ENTITY_LABEL_SEPARATOR}{entity['group']}"
            for entity in entities
            if "entity" in entity.keys() and "group" in entity.keys()
        )
        entity_names = entity_names_basic.union(entity_names_roles, entity_names_groups)

        # the USED_ENTITIES_KEY of an intent also contains the entity labels and the
        # concatenated entity labels with their corresponding roles and groups labels
        wanted_entities = set(intent_config.get(USED_ENTITIES_KEY, entity_names))

        return entity_names.intersection(wanted_entities)

    def _get_user_sub_state(
        self, tracker: "DialogueStateTracker"
    ) -> Dict[Text, Union[Text, Tuple[Text]]]:
        """Turns latest UserUttered event into a substate.

        The substate will contain intent, text, and entities (if any are present).

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

        # Filter entities based on intent config. We need to convert the set into a
        # tuple because sub_state will be later transformed into a frozenset (so it can
        # be hashed for deduplication).
        entities = tuple(
            self._get_featurized_entities(latest_message).intersection(
                set(sub_state.get(ENTITIES, ()))
            )
        )
        # Sort entities so that any derived state representation is consistent across
        # runs and invariant to the order in which the entities for an utterance are
        # listed in data files.
        entities = tuple(sorted(entities))

        if entities:
            sub_state[ENTITIES] = entities
        else:
            sub_state.pop(ENTITIES, None)

        return sub_state

    @staticmethod
    def _get_slots_sub_state(
        tracker: "DialogueStateTracker", omit_unset_slots: bool = False,
    ) -> Dict[Text, Union[Text, Tuple[float]]]:
        """Sets all set slots with the featurization of the stored value.

        Args:
            tracker: dialog state tracker containing the dialog so far
            omit_unset_slots: If `True` do not include the initial values of slots.

        Returns:
            a mapping of slot names to their featurization
        """
        slots = {}
        for slot_name, slot in tracker.slots.items():
            # If the slot doesn't influence conversations, slot.as_feature() will return
            # a result that evaluates to False, meaning that the slot shouldn't be
            # included in featurised sub-states.
            # Note that this condition checks if the slot itself is None. An unset slot
            # will be a Slot object and its `value` attribute will be None.
            if slot is not None and slot.as_feature():
                if omit_unset_slots and not slot.has_been_set:
                    continue
                if slot.value == rasa.shared.core.constants.SHOULD_NOT_BE_SET:
                    slots[slot_name] = rasa.shared.core.constants.SHOULD_NOT_BE_SET
                elif any(slot.as_feature()):
                    # Only include slot in featurised sub-state if the slot is not
                    # unset, i.e. is set to some actual value and has been successfully
                    # featurized, and hence has at least one non-zero feature.
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
        active_loop: Optional[Text] = tracker.active_loop.get(
            rasa.shared.core.constants.LOOP_NAME
        )
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

    def get_active_state(
        self, tracker: "DialogueStateTracker", omit_unset_slots: bool = False,
    ) -> State:
        """Given a dialogue tracker, makes a representation of current dialogue state.

        Args:
            tracker: dialog state tracker containing the dialog so far
            omit_unset_slots: If `True` do not include the initial values of slots.

        Returns:
            A representation of the dialogue's current state.
        """
        state = {
            rasa.shared.core.constants.USER: self._get_user_sub_state(tracker),
            rasa.shared.core.constants.SLOTS: self._get_slots_sub_state(
                tracker, omit_unset_slots=omit_unset_slots
            ),
            rasa.shared.core.constants.PREVIOUS_ACTION: self._get_prev_action_sub_state(
                tracker
            ),
            rasa.shared.core.constants.ACTIVE_LOOP: self._get_active_loop_sub_state(
                tracker
            ),
        }
        return self._clean_state(state)

    @staticmethod
    def _remove_rule_only_features(
        state: State, rule_only_data: Optional[Dict[Text, Any]],
    ) -> None:
        if not rule_only_data:
            return

        rule_only_slots = rule_only_data.get(
            rasa.shared.core.constants.RULE_ONLY_SLOTS, []
        )
        rule_only_loops = rule_only_data.get(
            rasa.shared.core.constants.RULE_ONLY_LOOPS, []
        )

        # remove slots which only occur in rules but not in stories
        if rule_only_slots:
            for slot in rule_only_slots:
                state.get(rasa.shared.core.constants.SLOTS, {}).pop(slot, None)
        # remove active loop which only occur in rules but not in stories
        if (
            rule_only_loops
            and state.get(rasa.shared.core.constants.ACTIVE_LOOP, {}).get(
                rasa.shared.core.constants.LOOP_NAME
            )
            in rule_only_loops
        ):
            del state[rasa.shared.core.constants.ACTIVE_LOOP]

    @staticmethod
    def _substitute_rule_only_user_input(state: State, last_ml_state: State) -> None:
        if not rasa.shared.core.trackers.is_prev_action_listen_in_state(state):
            if not last_ml_state.get(rasa.shared.core.constants.USER) and state.get(
                rasa.shared.core.constants.USER
            ):
                del state[rasa.shared.core.constants.USER]
            elif last_ml_state.get(rasa.shared.core.constants.USER):
                state[rasa.shared.core.constants.USER] = last_ml_state[
                    rasa.shared.core.constants.USER
                ]

    def states_for_tracker_history(
        self,
        tracker: "DialogueStateTracker",
        omit_unset_slots: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[State]:
        """List of states for each state of the trackers history.

        Args:
            tracker: Dialogue state tracker containing the dialogue so far.
            omit_unset_slots: If `True` do not include the initial values of slots.
            ignore_rule_only_turns: If True ignore dialogue turns that are present
                only in rules.
            rule_only_data: Slots and loops,
                which only occur in rules but not in stories.

        Return:
            A list of states.
        """
        states = []
        last_ml_action_sub_state = None
        turn_was_hidden = False
        for tr, hide_rule_turn in tracker.generate_all_prior_trackers():
            if ignore_rule_only_turns:
                # remember previous ml action based on the last non hidden turn
                # we need this to override previous action in the ml state
                if not turn_was_hidden:
                    last_ml_action_sub_state = self._get_prev_action_sub_state(tr)

                # followup action or happy path loop prediction
                # don't change the fact whether dialogue turn should be hidden
                if (
                    not tr.followup_action
                    and not tr.latest_action_name == tr.active_loop_name
                ):
                    turn_was_hidden = hide_rule_turn

                if turn_was_hidden:
                    continue

            state = self.get_active_state(tr, omit_unset_slots=omit_unset_slots)

            if ignore_rule_only_turns:
                # clean state from only rule features
                self._remove_rule_only_features(state, rule_only_data)
                # make sure user input is the same as for previous state
                # for non action_listen turns
                if states:
                    self._substitute_rule_only_user_input(state, states[-1])
                # substitute previous rule action with last_ml_action_sub_state
                if last_ml_action_sub_state:
                    state[
                        rasa.shared.core.constants.PREVIOUS_ACTION
                    ] = last_ml_action_sub_state

            states.append(self._clean_state(state))

        return states

    def slots_for_entities(self, entities: List[Dict[Text, Any]]) -> List[SlotSet]:
        """Creates slot events for entities if auto-filling is enabled.

        Args:
            entities: The list of entities.

        Returns:
            A list of `SlotSet` events.
        """
        if self.store_entities_as_slots:
            slot_events = []
            for slot in self.slots:
                if slot.auto_fill:
                    matching_entities = [
                        entity.get("value")
                        for entity in entities
                        if entity.get("entity") == slot.name
                    ]
                    if matching_entities:
                        if slot.type_name == "list":
                            slot_events.append(SlotSet(slot.name, matching_entities))
                        else:
                            slot_events.append(
                                SlotSet(slot.name, matching_entities[-1])
                            )
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
        # Only persist slots defined by the user. We add the default slots on the
        # fly when loading the domain.
        return {slot.name: slot.persistence_info() for slot in self._user_slots}

    def as_dict(self) -> Dict[Text, Any]:
        """Return serialized `Domain`."""
        return {
            "config": {"store_entities_as_slots": self.store_entities_as_slots},
            SESSION_CONFIG_KEY: {
                SESSION_EXPIRATION_TIME_KEY: (
                    self.session_config.session_expiration_time
                ),
                CARRY_OVER_SLOTS_KEY: self.session_config.carry_over_slots,
            },
            KEY_INTENTS: self._transform_intents_for_file(),
            KEY_ENTITIES: self._transform_entities_for_file(),
            KEY_SLOTS: self._slot_definitions(),
            KEY_RESPONSES: self.responses,
            KEY_ACTIONS: self._custom_actions,
            KEY_FORMS: self.forms,
            KEY_E2E_ACTIONS: self.action_texts,
        }

    @staticmethod
    def get_responses_with_multilines(
        responses: Dict[Text, List[Dict[Text, Any]]]
    ) -> Dict[Text, List[Dict[Text, Any]]]:
        """Returns `responses` with preserved multilines in the `text` key.

        Args:
            responses: Original `responses`.

        Returns:
            `responses` with preserved multilines in the `text` key.
        """
        from ruamel.yaml.scalarstring import LiteralScalarString

        final_responses = responses.copy()
        for utter_action, examples in final_responses.items():
            for i, example in enumerate(examples):
                response_text = example.get(KEY_RESPONSES_TEXT, "")
                if not response_text or "\n" not in response_text:
                    continue
                # Has new lines, use `LiteralScalarString`
                final_responses[utter_action][i][
                    KEY_RESPONSES_TEXT
                ] = LiteralScalarString(response_text)

        return final_responses

    def _transform_intents_for_file(self) -> List[Union[Text, Dict[Text, Any]]]:
        """Transform intent properties for displaying or writing into a domain file.

        Internally, there is a property `used_entities` that lists all entities to be
        used. In domain files, `use_entities` or `ignore_entities` is used instead to
        list individual entities to ex- or include, because this is easier to read.

        Returns:
            The intent properties as they are used in domain files.
        """
        intent_properties = copy.deepcopy(self.intent_properties)
        sorted_intent_properties = sorted(intent_properties.items())
        intents_for_file = []

        for intent_name, intent_props in sorted_intent_properties:
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
                entities = list(use_entities)
                entities.sort()
                intent_props[USE_ENTITIES_KEY] = entities
            else:
                entities = list(ignore_entities)
                entities.sort()
                intent_props[IGNORE_ENTITIES_KEY] = entities
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
            key: val
            for key, val in domain_data.items()
            if val != {} and val != [] and val is not None
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
            KEY_TRAINING_DATA_FORMAT_VERSION: LATEST_TRAINING_DATA_FORMAT_VERSION
        }
        if clean_before_dump:
            domain_data.update(self.cleaned_domain())
        else:
            domain_data.update(self.as_dict())
        if domain_data.get(KEY_RESPONSES, {}):
            domain_data[KEY_RESPONSES] = self.get_responses_with_multilines(
                domain_data[KEY_RESPONSES]
            )

        return rasa.shared.utils.io.dump_obj_as_yaml_to_string(
            domain_data, should_preserve_key_order=True
        )

    def intent_config(self, intent_name: Text) -> Dict[Text, Any]:
        """Return the configuration for an intent."""
        return self.intent_properties.get(intent_name, {})

    @rasa.shared.utils.common.lazy_property
    def intents(self) -> List[Text]:
        """Returns sorted list of intents."""
        return sorted(self.intent_properties.keys())

    @property
    def _slots_for_domain_warnings(self) -> List[Text]:
        """Fetch names of slots that are used in domain warnings.

        Excludes slots which aren't featurized.
        """

        return [slot.name for slot in self._user_slots if slot.influence_conversation]

    @property
    def _actions_for_domain_warnings(self) -> List[Text]:
        """Fetch names of actions that are used in domain warnings.

        Includes user and form actions, but excludes those that are default actions.
        """

        return [
            action
            for action in self.user_actions_and_forms
            if action not in rasa.shared.core.constants.DEFAULT_ACTION_NAMES
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
    def _combine_with_responses(
        actions: List[Text], responses: Dict[Text, Any]
    ) -> List[Text]:
        """Combines actions with utter actions listed in responses section."""
        unique_utter_actions = [
            action for action in sorted(list(responses.keys())) if action not in actions
        ]
        return actions + unique_utter_actions

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
            action
            for action in user_actions
            if action not in rasa.shared.core.constants.DEFAULT_ACTION_NAMES
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
        named or a response is missing.
        """

        def get_duplicates(my_items: Iterable[Any]) -> List[Any]:
            """Returns a list of duplicate items in my_items."""
            return [
                item
                for item, count in collections.Counter(my_items).items()
                if count > 1
            ]

        def check_mappings(
            intent_properties: Dict[Text, Dict[Text, Union[bool, List]]]
        ) -> List[Tuple[Text, Text]]:
            """Checks whether intent-action mappings use valid action names or texts."""
            incorrect = []
            for intent, properties in intent_properties.items():
                if "triggers" in properties:
                    triggered_action = properties.get("triggers")
                    if triggered_action not in self.action_names_or_texts:
                        incorrect.append((intent, str(triggered_action)))
            return incorrect

        def get_exception_message(
            duplicates: Optional[List[Tuple[List[Text], Text]]] = None,
            mappings: List[Tuple[Text, Text]] = None,
        ) -> Text:
            """Return a message given a list of error locations."""

            message = ""
            if duplicates:
                message += get_duplicate_exception_message(duplicates)
            if mappings:
                if message:
                    message += "\n"
                message += get_mapping_exception_message(mappings)
            return message

        def get_mapping_exception_message(mappings: List[Tuple[Text, Text]]) -> Text:
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

        duplicate_actions = get_duplicates(self.action_names_or_texts)
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

    @property
    def utterances_for_response(self) -> Set[Text]:
        """Returns utterance set which should have a response.

        Will filter out utterances which are subintent (retrieval intent) types.
        eg. if actions have ['utter_chitchat', 'utter_chitchat/greet'], this
        will only return ['utter_chitchat/greet'] as only that will need a
        response.
        """
        utterances = set()
        subintent_parents = set()
        for action in self.action_names_or_texts:
            if not action.startswith(rasa.shared.constants.UTTER_PREFIX):
                continue
            action_parent_split = action.split(RESPONSE_IDENTIFIER_DELIMITER)
            if len(action_parent_split) == 2:
                action_parent = action_parent_split[0]
                subintent_parents.add(action_parent)
            utterances.add(action)
        return utterances - subintent_parents

    def check_missing_templates(self) -> None:
        """Warn user of utterance names which have no specified response."""
        rasa.shared.utils.io.raise_deprecation_warning(
            "The terminology 'template' is deprecated "
            "and replaced by 'response'. "
            "Please use `check_missing_responses` "
            "instead of `check_missing_templates`.",
            docs=f"{DOCS_URL_MIGRATION_GUIDE}#rasa-23-to-rasa-24",
        )
        self.check_missing_responses()

    def check_missing_responses(self) -> None:
        """Warn user of utterance names which have no specified response."""
        missing_responses = self.utterances_for_response - set(self.responses)

        for response in missing_responses:
            rasa.shared.utils.io.raise_warning(
                f"Action '{response}' is listed as a "
                f"response action in the domain file, but there is "
                f"no matching response defined. Please check your domain.",
                docs=DOCS_URL_RESPONSES,
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

        try:
            content = rasa.shared.utils.io.read_yaml_file(filename)
        except (RasaException, YamlSyntaxException):
            rasa.shared.utils.io.raise_warning(
                message=f"The file {filename} could not be loaded as domain file. "
                + "You can use https://yamlchecker.com/ to validate "
                + "the YAML syntax of your file.",
                category=UserWarning,
            )
            return False

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
        form = self.forms.get(form_name)
        if form:
            return form[REQUIRED_SLOTS_KEY]

        return {}


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
                f"{DOCS_URL_FORMS} for more information."
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
                f"{DOCS_URL_FORMS} for more information."
            )

        for required_key in required_keys:
            if mapping.get(required_key) is None:
                raise InvalidDomain(
                    f"You need to specify a value for the key "
                    f"'{required_key}' in the slot mapping of type '{mapping_type}' "
                    f"for slot '{slot_name}' in the form '{form_name}'. Please see "
                    f"{DOCS_URL_FORMS} for more information."
                )


def _validate_slot_mappings(forms: Union[Dict, List]) -> None:
    if isinstance(forms, list):
        if not all(isinstance(form_name, str) for form_name in forms):
            raise InvalidDomain(
                f"If you use the deprecated list syntax for forms, "
                f"all form names have to be strings. Please see "
                f"{DOCS_URL_FORMS} for more information."
            )

        return

    if not isinstance(forms, dict):
        raise InvalidDomain("Forms have to be specified as dictionary.")

    for form_name, form_data in forms.items():
        if form_data is None:
            continue

        if not isinstance(form_data, Dict):
            raise InvalidDomain(
                f"The contents of form '{form_name}' were specified "
                f"as '{type(form_data)}'. They need to be specified "
                f"as dictionary. Please see {DOCS_URL_FORMS} "
                f"for more information."
            )

        if IGNORED_INTENTS in form_data and REQUIRED_SLOTS_KEY not in form_data:
            raise InvalidDomain(
                f"If you use the `{IGNORED_INTENTS}` parameter in your form, then "
                f"the keyword `{REQUIRED_SLOTS_KEY}` should precede the definition "
                f"of your slot mappings. Please see {DOCS_URL_FORMS} "
                f"for more information."
            )

        if REQUIRED_SLOTS_KEY in form_data:
            slots = forms[form_name].get(REQUIRED_SLOTS_KEY)
        else:
            rasa.shared.utils.io.raise_deprecation_warning(
                f"The definition of slot mappings in your form "
                f"should always be preceded by the keyword `{REQUIRED_SLOTS_KEY}`. "
                f"The lack of this keyword will be deprecated in "
                f"Rasa Open Source 3.0.0. Please see {DOCS_URL_FORMS} "
                f"for more information.",
            )
            slots = form_data

        if not isinstance(slots, Dict):
            raise InvalidDomain(
                f"The slots for form '{form_name}' were specified "
                f"as '{type(slots)}'. They need to be specified "
                f"as dictionary. Please see {DOCS_URL_FORMS} "
                f"for more information."
            )

        for slot_name, slot_mappings in slots.items():
            if not isinstance(slot_mappings, list):
                raise InvalidDomain(
                    f"The slot mappings for slot '{slot_name}' in "
                    f"form '{form_name}' have type '{type(slot_mappings)}'. "
                    f"It is required to provide a list of slot "
                    f"mappings. Please see {DOCS_URL_FORMS} "
                    f"for more information."
                )
            for slot_mapping in slot_mappings:
                SlotMapping.validate(slot_mapping, form_name, slot_name)
