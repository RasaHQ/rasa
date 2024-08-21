import collections
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    MutableMapping,
    NamedTuple,
    NoReturn,
    Optional,
    Set,
    Text,
    Tuple,
    Union,
    cast,
)

import structlog
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

import rasa.shared.core.slot_mappings
import rasa.shared.utils.common
import rasa.shared.utils.io
from rasa.shared.constants import (
    DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION,
    DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES,
    DOCS_URL_DOMAINS,
    DOCS_URL_FORMS,
    DOCS_URL_RESPONSES,
    DOMAIN_SCHEMA_FILE,
    IGNORED_INTENTS,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
    REQUIRED_SLOTS_KEY,
    RESPONSE_CONDITION,
)
from rasa.shared.core.constants import (
    ACTION_SHOULD_SEND_DOMAIN,
    ACTIVE_LOOP,
    KNOWLEDGE_BASE_SLOT_NAMES,
    MAPPING_CONDITIONS,
    MAPPING_TYPE,
    SLOT_MAPPINGS,
    SlotMappingType,
)
from rasa.shared.core.events import SlotSet, UserUttered
from rasa.shared.core.slots import (
    AnySlot,
    CategoricalSlot,
    ListSlot,
    Slot,
    TextSlot,
)
from rasa.shared.exceptions import (
    RasaException,
    YamlException,
    YamlSyntaxException,
)
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_TYPE,
    INTENT_NAME_KEY,
    RESPONSE_IDENTIFIER_DELIMITER,
)
from rasa.shared.utils.cli import print_error_and_exit
from rasa.shared.utils.yaml import (
    KEY_TRAINING_DATA_FORMAT_VERSION,
    dump_obj_as_yaml_to_string,
    read_yaml,
    read_yaml_file,
    validate_raw_yaml_using_schema_file_with_responses,
    validate_training_data_format_version,
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
ENTITY_FEATURIZATION_KEY = "influence_conversation"

KEY_SLOTS = "slots"
KEY_INTENTS = "intents"
KEY_ENTITIES = "entities"
KEY_RESPONSES = "responses"
KEY_ACTIONS = "actions"
KEY_FORMS = "forms"
KEY_E2E_ACTIONS = "e2e_actions"
KEY_RESPONSES_TEXT = "text"
KEY_RESPONSES_IMAGE = "image"
KEY_RESPONSES_CUSTOM = "custom"
KEY_RESPONSES_BUTTONS = "buttons"
KEY_RESPONSES_ATTACHMENT = "attachment"
KEY_RESPONSES_QUICK_REPLIES = "quick_replies"

RESPONSE_KEYS_TO_INTERPOLATE = [
    KEY_RESPONSES_TEXT,
    KEY_RESPONSES_IMAGE,
    KEY_RESPONSES_CUSTOM,
    KEY_RESPONSES_BUTTONS,
    KEY_RESPONSES_ATTACHMENT,
    KEY_RESPONSES_QUICK_REPLIES,
]

ALL_DOMAIN_KEYS = [
    KEY_SLOTS,
    KEY_FORMS,
    KEY_ACTIONS,
    KEY_ENTITIES,
    KEY_INTENTS,
    KEY_RESPONSES,
    KEY_E2E_ACTIONS,
    SESSION_CONFIG_KEY,
]

PREV_PREFIX = "prev_"

# State is a dictionary with keys (USER, PREVIOUS_ACTION, SLOTS, ACTIVE_LOOP)
# representing the origin of a SubState;
# the values are SubStates, that contain the information needed for featurization
SubStateValue = Union[Text, Tuple[Union[float, Text], ...]]
SubState = MutableMapping[Text, SubStateValue]
State = Dict[Text, SubState]

structlogger = structlog.get_logger(__name__)


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

    def as_dict(self) -> Dict:
        """Return serialized `SessionConfig`."""
        return {
            "session_expiration_time": self.session_expiration_time,
            "carry_over_slots_to_new_session": self.carry_over_slots,
        }


@dataclass
class EntityProperties:
    """Class for keeping track of the properties of entities in the domain."""

    entities: List[Text]
    roles: Dict[Text, List[Text]]
    groups: Dict[Text, List[Text]]
    default_ignored_entities: List[Text]


class Domain:
    """The domain specifies the universe in which the bot's policy acts.

    A Domain subclass provides the actions the bot can take, the intents
    and entities it can recognise.
    """

    @classmethod
    def empty(cls) -> "Domain":
        """Returns empty Domain."""
        return Domain.from_dict({})

    @classmethod
    def load(cls, paths: Union[List[Union[Path, Text]], Text, Path]) -> "Domain":
        """Returns loaded Domain after merging all domain files."""
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
        """Loads the `Domain` from a path."""
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
            validate_raw_yaml_using_schema_file_with_responses(yaml, DOMAIN_SCHEMA_FILE)

            data = read_yaml(yaml)
            if not validate_training_data_format_version(data, original_filename):
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

        domain_slots = data.get(KEY_SLOTS, {})
        if domain_slots:
            rasa.shared.core.slot_mappings.validate_slot_mappings(domain_slots)
        slots = cls.collect_slots(domain_slots)
        domain_actions = data.get(KEY_ACTIONS, [])
        actions = cls._collect_action_names(domain_actions)

        additional_arguments = {
            **data.get("config", {}),
            "actions_which_explicitly_need_domain": cls._collect_actions_which_explicitly_need_domain(  # noqa: E501
                domain_actions
            ),
        }
        session_config = cls._get_session_config(data.get(SESSION_CONFIG_KEY, {}))
        intents = data.get(KEY_INTENTS, {})

        forms = data.get(KEY_FORMS, {})
        _validate_forms(forms)

        return cls(
            intents=intents,
            entities=data.get(KEY_ENTITIES, {}),
            slots=slots,
            responses=responses,
            action_names=actions,
            forms=data.get(KEY_FORMS, {}),
            data=Domain._cleaned_data(data),
            action_texts=data.get(KEY_E2E_ACTIONS, []),
            session_config=session_config,
            **additional_arguments,
        )

    @staticmethod
    def _get_session_config(session_config: Dict) -> SessionConfig:
        session_expiration_time_min = session_config.get(SESSION_EXPIRATION_TIME_KEY)

        if session_expiration_time_min is None:
            session_expiration_time_min = DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES

        carry_over_slots = session_config.get(
            CARRY_OVER_SLOTS_KEY, DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION
        )

        return SessionConfig(session_expiration_time_min, carry_over_slots)

    @classmethod
    def from_directory(cls, path: Text) -> "Domain":
        """Loads and merges multiple domain files recursively from a directory tree."""
        combined: Dict[Text, Any] = {}
        duplicates: List[Dict[Text, List[Text]]] = []

        for root, _, files in os.walk(path, followlinks=True):
            for file in files:
                full_path = os.path.join(root, file)
                if not Domain.is_domain_file(full_path):
                    continue

                # does the validation here only
                _ = Domain.from_file(full_path)

                other_dict = read_yaml(rasa.shared.utils.io.read_file(full_path))
                combined = Domain.merge_domain_dicts(other_dict, combined)
                duplicates.append(combined.pop("duplicates", {}))

        Domain._handle_duplicates_from_multiple_files(duplicates)

        domain = Domain.from_dict(combined)
        return domain

    @staticmethod
    def _handle_duplicates_from_multiple_files(
        duplicates_from_multiple_files: List[Dict[Text, List[Text]]],
    ) -> None:
        combined_duplicates: Dict[Text, List[Text]] = collections.defaultdict(list)

        for duplicates in duplicates_from_multiple_files:
            duplicates = rasa.shared.utils.common.clean_duplicates(duplicates)

            for key in duplicates.keys():
                combined_duplicates[key].extend(duplicates[key])

            # handle duplicated responses by raising an error
            duplicated_responses = combined_duplicates.pop(KEY_RESPONSES, [])
            Domain._handle_duplicate_responses(duplicated_responses)

            # warn about other duplicates
            warn_about_duplicates_found_during_domain_merging(combined_duplicates)

    @staticmethod
    def _handle_duplicate_responses(response_duplicates: List[Text]) -> None:
        if response_duplicates:
            for response in response_duplicates:
                structlogger.error(
                    "domain.duplicate_response",
                    response=response,
                    event_info=(
                        f"Response '{response}' is defined in multiple domains. "
                        f"Please make sure this response is only defined in one domain."
                    ),
                )
            print_error_and_exit(
                "Unable to merge domains due to duplicate responses in domain."
            )

    def merge(
        self,
        domain: Optional["Domain"],
        override: bool = False,
        ignore_warnings_about_duplicates: bool = False,
    ) -> "Domain":
        """Merges this domain dict with another one, combining their attributes.

        This method merges domain dicts, and ensures all attributes (like ``intents``,
        ``entities``, and ``actions``) are known to the Domain when the
        object is created.

        List attributes like ``intents`` and ``actions`` are deduped
        and merged. Single attributes are taken from `domain1` unless
        override is `True`, in which case they are taken from `domain2`.
        """
        if not domain or domain.is_empty():
            return self

        if self.is_empty():
            return domain

        combined = self.__class__.merge_domain_dicts(
            domain.as_dict(), self.as_dict(), override
        )

        duplicates = combined.pop("duplicates", {})

        if not ignore_warnings_about_duplicates:
            warn_about_duplicates_found_during_domain_merging(duplicates)

        return Domain.from_dict(combined)

    @staticmethod
    def merge_domain_dicts(
        domain_dict: Dict,
        combined: Dict,
        override: bool = False,
    ) -> Dict:
        """Combines two domain dictionaries."""
        if not domain_dict:
            return combined

        if not combined:
            return domain_dict

        if override:
            config = domain_dict.get("config", {})
            for key, val in config.items():
                combined["config"][key] = val

        if (
            override
            or combined.get(SESSION_CONFIG_KEY) == SessionConfig.default().as_dict()
            or combined.get(SESSION_CONFIG_KEY) is None
        ) and domain_dict.get(SESSION_CONFIG_KEY) not in [
            None,
            SessionConfig.default().as_dict(),
        ]:
            combined[SESSION_CONFIG_KEY] = domain_dict[SESSION_CONFIG_KEY]

        # remove existing forms from new actions
        for form in combined.get(KEY_FORMS, []):
            if form in domain_dict.get(KEY_ACTIONS, []):
                domain_dict[KEY_ACTIONS].remove(form)

        duplicates: Dict[Text, List[Text]] = {}

        merge_func_mappings: Dict[Text, Callable[..., Any]] = {
            KEY_INTENTS: rasa.shared.utils.common.merge_lists_of_dicts,
            KEY_ENTITIES: rasa.shared.utils.common.merge_lists_of_dicts,
            KEY_ACTIONS: rasa.shared.utils.common.merge_lists_of_dicts,
            KEY_E2E_ACTIONS: rasa.shared.utils.common.merge_lists,
            KEY_FORMS: rasa.shared.utils.common.merge_dicts,
            KEY_RESPONSES: rasa.shared.utils.common.merge_dicts,
            KEY_SLOTS: rasa.shared.utils.common.merge_dicts,
        }

        for key, merge_func in merge_func_mappings.items():
            duplicates[key] = rasa.shared.utils.common.extract_duplicates(
                combined.get(key, []), domain_dict.get(key, [])
            )

            default: Union[List[Any], Dict[Text, Any]] = (
                {} if merge_func == rasa.shared.utils.common.merge_dicts else []
            )

            combined[key] = merge_func(
                combined.get(key, default), domain_dict.get(key, default), override
            )

        if duplicates:
            combined.update({"duplicates": duplicates})

        return combined

    def _preprocess_domain_dict(
        self,
        data: Dict,
        store_entities_as_slots: bool,
        session_config: SessionConfig,
    ) -> Dict:
        data = self._add_default_keys_to_domain_dict(
            data,
            store_entities_as_slots,
            session_config,
        )
        data = self._sanitize_intents_in_domain_dict(data)

        return data

    @staticmethod
    def _add_default_keys_to_domain_dict(
        data: Dict,
        store_entities_as_slots: bool,
        session_config: SessionConfig,
    ) -> Dict:
        # add the config, session_config and training data version defaults
        # if not included in the original domain dict
        if "config" not in data and not store_entities_as_slots:
            data.update(
                {"config": {"store_entities_as_slots": store_entities_as_slots}}
            )

        if SESSION_CONFIG_KEY not in data:
            data.update(
                {
                    SESSION_CONFIG_KEY: {
                        SESSION_EXPIRATION_TIME_KEY: (
                            session_config.session_expiration_time
                        ),
                        CARRY_OVER_SLOTS_KEY: session_config.carry_over_slots,
                    }
                }
            )

        if KEY_TRAINING_DATA_FORMAT_VERSION not in data:
            data.update(
                {
                    KEY_TRAINING_DATA_FORMAT_VERSION: DoubleQuotedScalarString(
                        LATEST_TRAINING_DATA_FORMAT_VERSION
                    )
                }
            )

        return data

    @staticmethod
    def _reset_intent_flags(intent: Dict[Text, Any]) -> None:
        for intent_property in intent.values():
            if (
                USE_ENTITIES_KEY in intent_property.keys()
                and not intent_property[USE_ENTITIES_KEY]
            ):
                intent_property[USE_ENTITIES_KEY] = []
            if (
                IGNORE_ENTITIES_KEY in intent_property.keys()
                and not intent_property[IGNORE_ENTITIES_KEY]
            ):
                intent_property[IGNORE_ENTITIES_KEY] = []

    @staticmethod
    def _sanitize_intents_in_domain_dict(data: Dict[Text, Any]) -> Dict[Text, Any]:
        if not data.get(KEY_INTENTS):
            return data

        for intent in data.get(KEY_INTENTS, []):
            if isinstance(intent, dict):
                Domain._reset_intent_flags(intent)

        data[KEY_INTENTS] = Domain._sort_intent_names_alphabetical_order(
            intents=data.get(KEY_INTENTS)
        )

        return data

    @staticmethod
    def collect_slots(slot_dict: Dict[Text, Any]) -> List[Slot]:
        """Collects a list of slots from a dictionary."""
        slots = []
        # make a copy to not alter the input dictionary
        slot_dict = copy.deepcopy(slot_dict)
        # Don't sort the slots, see https://github.com/RasaHQ/rasa-x/issues/3900
        for slot_name in slot_dict:
            slot_type = slot_dict[slot_name].pop("type", None)
            slot_class = Slot.resolve_by_type(slot_type)

            if SLOT_MAPPINGS not in slot_dict[slot_name]:
                structlogger.debug(
                    "domain.collect_slots.no_mappings_defined",
                    event_info=(
                        f"Slot '{slot_name}' has no mappings defined. "
                        f"Assigning the default FROM_LLM slot mapping."
                    ),
                )
                slot_dict[slot_name][SLOT_MAPPINGS] = [
                    {MAPPING_TYPE: SlotMappingType.FROM_LLM.value}
                ]

            slot = slot_class(slot_name, **slot_dict[slot_name])
            slots.append(slot)
        return slots

    @staticmethod
    def _transform_intent_properties_for_internal_use(
        intent: Dict[Text, Any], entity_properties: EntityProperties
    ) -> Dict[Text, Any]:
        """Transforms the intent's parameters in a format suitable for internal use.

        When an intent is retrieved from the `domain.yml` file, it contains two
        parameters, the `use_entities` and the `ignore_entities` parameter.
        With the values of these two parameters the Domain class is updated, a new
        parameter is added to the intent called `used_entities` and the two
        previous parameters are deleted. This happens because internally only the
        parameter `used_entities` is needed to list all the entities that should be
        used for this intent.

        Args:
            intent: The intent as retrieved from the `domain.yml` file thus having two
                parameters, the `use_entities` and the `ignore_entities` parameter.
            entity_properties: Entity properties as provided by the domain file.

        Returns:
            The intent with the new format thus having only one parameter called
            `used_entities` since this is the expected format of the intent
            when used internally.
        """
        name, properties = next(iter(intent.items()))

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

        properties.setdefault(
            IGNORE_ENTITIES_KEY, entity_properties.default_ignored_entities
        )
        if not properties[USE_ENTITIES_KEY]:  # this covers False, None and []
            properties[USE_ENTITIES_KEY] = []

        # `use_entities` is either a list of explicitly included entities
        # or `True` if all should be included
        # if the listed entities have a role or group label, concatenate the entity
        # label with the corresponding role or group label to make sure roles and
        # groups can also influence the dialogue predictions
        if properties[USE_ENTITIES_KEY] is True:
            included_entities = set(entity_properties.entities) - set(
                entity_properties.default_ignored_entities
            )
            included_entities.update(
                Domain.concatenate_entity_labels(entity_properties.roles)
            )
            included_entities.update(
                Domain.concatenate_entity_labels(entity_properties.groups)
            )
        else:
            included_entities = set(properties[USE_ENTITIES_KEY])
            for entity in list(included_entities):
                included_entities.update(
                    Domain.concatenate_entity_labels(entity_properties.roles, entity)
                )
                included_entities.update(
                    Domain.concatenate_entity_labels(entity_properties.groups, entity)
                )
        excluded_entities = set(properties[IGNORE_ENTITIES_KEY])
        for entity in list(excluded_entities):
            excluded_entities.update(
                Domain.concatenate_entity_labels(entity_properties.roles, entity)
            )
            excluded_entities.update(
                Domain.concatenate_entity_labels(entity_properties.groups, entity)
            )
        used_entities = list(included_entities - excluded_entities)
        used_entities.sort()

        # Only print warning for ambiguous configurations if entities were included
        # explicitly.
        explicitly_included = isinstance(properties[USE_ENTITIES_KEY], list)
        ambiguous_entities = included_entities.intersection(excluded_entities)
        if explicitly_included and ambiguous_entities:
            structlogger.warning(
                "domain.ambiguous_entities",
                intent=name,
                entities=ambiguous_entities,
                event_info=(
                    f"Entities: '{ambiguous_entities}' are "
                    f"explicitly included and excluded for "
                    f"intent '{name}'. Excluding takes precedence "
                    f"in this case. Please resolve that ambiguity."
                ),
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
    ) -> EntityProperties:
        """Get entity properties for a domain from what is provided by a domain file.

        Args:
            domain_entities: The entities as provided by a domain file.

        Returns:
            An instance of EntityProperties.
        """
        entity_properties = EntityProperties([], {}, {}, [])
        for entity in domain_entities:
            if isinstance(entity, str):
                entity_properties.entities.append(entity)
            elif isinstance(entity, dict):
                for _entity, sub_labels in entity.items():
                    entity_properties.entities.append(_entity)
                    if sub_labels:
                        if ENTITY_ROLES_KEY in sub_labels:
                            entity_properties.roles[_entity] = sub_labels[
                                ENTITY_ROLES_KEY
                            ]
                        if ENTITY_GROUPS_KEY in sub_labels:
                            entity_properties.groups[_entity] = sub_labels[
                                ENTITY_GROUPS_KEY
                            ]
                        if (
                            ENTITY_FEATURIZATION_KEY in sub_labels
                            and sub_labels[ENTITY_FEATURIZATION_KEY] is False
                        ):
                            entity_properties.default_ignored_entities.append(_entity)
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

        return entity_properties

    @classmethod
    def collect_intent_properties(
        cls,
        intents: List[Union[Text, Dict[Text, Any]]],
        entity_properties: EntityProperties,
    ) -> Dict[Text, Dict[Text, Union[bool, List]]]:
        """Get intent properties for a domain from what is provided by a domain file.

        Args:
            intents: The intents as provided by a domain file.
            entity_properties: Entity properties as provided by the domain file.

        Returns:
            The intent properties to be stored in the domain.
        """
        # make a copy to not alter the input argument
        intents = copy.deepcopy(intents)
        intent_properties: Dict[Text, Any] = {}
        duplicates = set()

        for intent in intents:
            intent_name, properties = cls._intent_properties(intent, entity_properties)

            if intent_name in intent_properties.keys():
                duplicates.add(intent_name)

            intent_properties.update(properties)

        if duplicates:
            raise InvalidDomain(
                f"Intents are not unique! Found multiple intents "
                f"with name(s) {sorted(duplicates)}. "
                f"Either rename or remove the duplicate ones."
            )

        cls._add_default_intents(intent_properties, entity_properties)

        return intent_properties

    @classmethod
    def _intent_properties(
        cls, intent: Union[Text, Dict[Text, Any]], entity_properties: EntityProperties
    ) -> Tuple[Text, Dict[Text, Any]]:
        if not isinstance(intent, dict):
            intent_name = intent
            intent = {
                intent_name: {
                    USE_ENTITIES_KEY: True,
                    IGNORE_ENTITIES_KEY: entity_properties.default_ignored_entities,
                }
            }
        else:
            intent_name = next(iter(intent.keys()))

        return (
            intent_name,
            cls._transform_intent_properties_for_internal_use(
                intent, entity_properties
            ),
        )

    @classmethod
    def _add_default_intents(
        cls,
        intent_properties: Dict[Text, Dict[Text, Union[bool, List]]],
        entity_properties: EntityProperties,
    ) -> None:
        for intent_name in rasa.shared.core.constants.DEFAULT_INTENTS:
            if intent_name not in intent_properties:
                _, properties = cls._intent_properties(intent_name, entity_properties)
                intent_properties.update(properties)

    def __init__(
        self,
        intents: Union[Set[Text], List[Text], List[Dict[Text, Any]]],
        entities: List[Union[Text, Dict[Text, Any]]],
        slots: List[Slot],
        responses: Dict[Text, List[Dict[Text, Any]]],
        action_names: List[Text],
        forms: Union[Dict[Text, Any], List[Text]],
        data: Dict,
        action_texts: Optional[List[Text]] = None,
        store_entities_as_slots: bool = True,
        session_config: SessionConfig = SessionConfig.default(),
        **kwargs: Any,
    ) -> None:
        """Create a `Domain`.

        Args:
            intents: Intent labels.
            entities: The names of entities which might be present in user messages.
            slots: Slots to store information during the conversation.
            responses: Bot responses. If an action with the same name is executed, it
                will send the matching response to the user.
            action_names: Names of custom actions.
            forms: Form names and their slot mappings.
            data: original domain dict representation.
            action_texts: End-to-End bot utterances from end-to-end stories.
            store_entities_as_slots: If `True` Rasa will automatically create `SlotSet`
                events for entities if there are slots with the same name as the entity.
            session_config: Configuration for conversation sessions. Conversations are
                restarted at the end of a session.
            kwargs: Additional arguments.
        """
        self.entity_properties = self.collect_entity_properties(entities)
        self.intent_properties = self.collect_intent_properties(
            intents, self.entity_properties
        )
        self.overridden_default_intents = self._collect_overridden_default_intents(
            intents
        )

        self.form_names, self.forms, overridden_form_actions = self._initialize_forms(
            forms
        )

        action_names += overridden_form_actions

        self.responses = responses

        self.action_texts = action_texts if action_texts is not None else []

        data_copy = copy.deepcopy(data)
        self._data = self._preprocess_domain_dict(
            data_copy,
            store_entities_as_slots,
            session_config,
        )

        self.session_config = session_config

        self._custom_actions = action_names
        self._actions_which_explicitly_need_domain = (
            kwargs.get("actions_which_explicitly_need_domain") or []
        )

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
        intents: Union[Set[Text], List[Text], List[Dict[Text, Any]]],
    ) -> List[Text]:
        """Collects the default intents overridden by the user.

        Args:
            intents: User-provided intents.

        Returns:
            User-defined intents that are default intents.
        """
        intent_names: Set[Text] = {
            next(iter(intent.keys())) if isinstance(intent, dict) else intent
            for intent in intents
        }
        return sorted(
            intent_names.intersection(set(rasa.shared.core.constants.DEFAULT_INTENTS))
        )

    @staticmethod
    def _initialize_forms(
        forms: Dict[Text, Any],
    ) -> Tuple[List[Text], Dict[Text, Any], List[Text]]:
        """Retrieves the initial values for the Domain's form fields.

        Args:
            forms: Parsed content of the `forms` section in the domain.

        Returns:
            The form names, a mapping of form names and required slots, and custom
            actions.
            Returning custom actions for each forms means that Rasa Pro should
            not use the default `FormAction` for the forms, but rather a custom action
            for it. This can e.g. be used to run the deprecated Rasa Open Source 1
            `FormAction` which is implemented in the Rasa SDK.
        """
        for form_name, form_data in forms.items():
            if form_data is not None and REQUIRED_SLOTS_KEY not in form_data:
                forms[form_name] = {REQUIRED_SLOTS_KEY: form_data}
        return list(forms.keys()), forms, []

    def __hash__(self) -> int:
        """Returns a unique hash for the domain."""
        return int(self.fingerprint(), 16)

    @rasa.shared.utils.common.cached_method
    def fingerprint(self) -> Text:
        """Returns a unique hash for the domain which is stable across python runs.

        Returns:
            fingerprint of the domain
        """
        self_as_dict = self.as_dict()
        transformed_intents: List[Text] = []
        for intent in self_as_dict.get(KEY_INTENTS, []):
            if isinstance(intent, dict):
                transformed_intents.append(*intent.keys())
            elif isinstance(intent, str):
                transformed_intents.append(intent)

        self_as_dict[KEY_INTENTS] = sorted(transformed_intents)
        self_as_dict[KEY_ACTIONS] = self.action_names_or_texts
        return rasa.shared.utils.io.get_dictionary_fingerprint(self_as_dict)

    @staticmethod
    def _sort_intent_names_alphabetical_order(
        intents: List[Union[Text, Dict]],
    ) -> List[Union[Text, Dict]]:
        def sort(elem: Union[Text, Dict]) -> Union[Text, Dict]:
            if isinstance(elem, dict):
                return next(iter(elem.keys()))
            elif isinstance(elem, str):
                return elem

        sorted_intents = sorted(intents, key=sort)
        return sorted_intents

    @rasa.shared.utils.common.lazy_property
    def user_actions_and_forms(self) -> List[Text]:
        """Returns combination of user actions and forms."""
        return self.user_actions + self.form_names

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

    @staticmethod
    def is_retrieval_intent_response(
        response: Tuple[Text, List[Dict[Text, Any]]],
    ) -> bool:
        """Check if the response is for a retrieval intent.

        These responses have a `/` symbol in their name. Use that to filter them from
        the rest.
        """
        return RESPONSE_IDENTIFIER_DELIMITER in response[0]

    def _add_default_slots(self) -> None:
        """Sets up the default slots and slot values for the domain."""
        self._add_requested_slot()
        self._add_flow_slots()
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

    def _add_flow_slots(self) -> None:
        """Adds the slots needed for the conversation flows."""
        from rasa.shared.core.constants import FLOW_SLOT_NAMES

        slot_names = [slot.name for slot in self.slots]

        for flow_slot in FLOW_SLOT_NAMES:
            if flow_slot not in slot_names:
                self.slots.append(
                    AnySlot(
                        flow_slot,
                        mappings=[],
                        influence_conversation=False,
                        is_builtin=True,
                    )
                )
            else:
                # TODO: in the future we need to prevent this entirely.
                structlogger.error(
                    "domain.add_flow_slots.slot_reserved_for_internal_usage",
                    event_info=(
                        f"Slot {flow_slot} is reserved for Rasa internal usage, "
                        f"but it already exists. This might lead to bad outcomes."
                    ),
                )

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
                    mappings=[],
                    influence_conversation=False,
                    is_builtin=True,
                )
            )

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
            structlogger.warning(
                "domain.add_knowledge_base_slots.use_of_experimental_feature",
                event_info=(
                    "You are using an experimental feature: Action '{}'!".format(
                        rasa.shared.core.constants.DEFAULT_KNOWLEDGE_BASE_ACTION
                    )
                ),
            )
            slot_names = [slot.name for slot in self.slots]
            for slot in KNOWLEDGE_BASE_SLOT_NAMES:
                if slot not in slot_names:
                    self.slots.append(
                        TextSlot(
                            slot,
                            mappings=[],
                            influence_conversation=False,
                            is_builtin=True,
                        )
                    )

    def _add_session_metadata_slot(self) -> None:
        self.slots.append(
            AnySlot(
                rasa.shared.core.constants.SESSION_START_METADATA_SLOT,
                mappings=[],
                is_builtin=True,
            )
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
        entity_states.extend(
            Domain.concatenate_entity_labels(self.entity_properties.roles)
        )
        entity_states.extend(
            Domain.concatenate_entity_labels(self.entity_properties.groups)
        )

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

    def _get_user_sub_state(self, tracker: "DialogueStateTracker") -> SubState:
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

        sub_state = cast(SubState, latest_message.as_sub_state())

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
        tracker: "DialogueStateTracker", omit_unset_slots: bool = False
    ) -> SubState:
        """Sets all set slots with the featurization of the stored value.

        Args:
            tracker: dialog state tracker containing the dialog so far
            omit_unset_slots: If `True` do not include the initial values of slots.

        Returns:
            a mapping of slot names to their featurization
        """
        slots: SubState = {}
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
    ) -> Optional[Dict[Text, Text]]:
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
    ) -> Dict[Text, Optional[Text]]:
        """Turn tracker's active loop into a state name.

        Args:
            tracker: dialog state tracker containing the dialog so far
        Returns:
            a dictionary mapping "name" to active loop name if present
        """
        # we don't use tracker.active_loop_name
        # because we need to keep should_not_be_set
        if tracker.active_loop:
            return {rasa.shared.core.constants.LOOP_NAME: tracker.active_loop.name}
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
        self, tracker: "DialogueStateTracker", omit_unset_slots: bool = False
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
        state: State, rule_only_data: Optional[Dict[Text, Any]]
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
        states: List[State] = []
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
                    # FIXME: better type annotation for `State` would require
                    # a larger refactoring (e.g. switch to dataclass)
                    state[rasa.shared.core.constants.PREVIOUS_ACTION] = cast(
                        SubState,
                        last_ml_action_sub_state,
                    )

            states.append(self._clean_state(state))

        return states

    def slots_for_entities(self, entities: List[Dict[Text, Any]]) -> List[SlotSet]:
        """Creates slot events for entities if from_entity mapping matches.

        Args:
            entities: The list of entities.

        Returns:
            A list of `SlotSet` events.
        """
        if self.store_entities_as_slots:
            slot_events = []

            for slot in self.slots:
                matching_entities = []

                for mapping in slot.mappings:
                    mapping_conditions = mapping.get(MAPPING_CONDITIONS)
                    if mapping[MAPPING_TYPE] != str(SlotMappingType.FROM_ENTITY) or (
                        mapping_conditions
                        and mapping_conditions[0].get(ACTIVE_LOOP) is not None
                    ):
                        continue

                    for entity in entities:
                        if (
                            entity.get(ENTITY_ATTRIBUTE_TYPE)
                            == mapping.get(ENTITY_ATTRIBUTE_TYPE)
                            and entity.get(ENTITY_ATTRIBUTE_ROLE)
                            == mapping.get(ENTITY_ATTRIBUTE_ROLE)
                            and entity.get(ENTITY_ATTRIBUTE_GROUP)
                            == mapping.get(ENTITY_ATTRIBUTE_GROUP)
                        ):
                            matching_entities.append(entity.get("value"))

                if matching_entities:
                    if isinstance(slot, ListSlot):
                        slot_events.append(SlotSet(slot.name, matching_entities))
                    else:
                        slot_events.append(SlotSet(slot.name, matching_entities[-1]))

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

    def as_dict(self) -> Dict[Text, Any]:
        """Return serialized `Domain`."""
        return self._data

    @staticmethod
    def get_responses_with_multilines(
        responses: Dict[Text, List[Dict[Text, Any]]],
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
                final_responses[utter_action][i][KEY_RESPONSES_TEXT] = (
                    LiteralScalarString(response_text)
                )

        return final_responses

    @staticmethod
    def _cleaned_data(data: Dict[Text, Any]) -> Dict[Text, Any]:
        """Remove empty and redundant keys from merged domain dict.

        Returns:
            A cleaned dictionary version of the domain.
        """
        return {
            key: val
            for key, val in data.items()
            if val != {} and val != [] and val is not None
        }

    def persist(self, filename: Union[Text, Path]) -> None:
        """Write domain to a file."""
        as_yaml = self.as_yaml()
        rasa.shared.utils.io.write_text_file(as_yaml, filename)

    def as_yaml(self) -> Text:
        """Dump the `Domain` object as a YAML string.

        This function preserves the orders of the keys in the domain.

        Returns:
            A string in YAML format representing the domain.
        """
        # setting the `version` key first so that it appears at the top of YAML files
        # thanks to the `should_preserve_key_order` argument
        # of `dump_obj_as_yaml_to_string`
        domain_data: Dict[Text, Any] = {
            KEY_TRAINING_DATA_FORMAT_VERSION: DoubleQuotedScalarString(
                LATEST_TRAINING_DATA_FORMAT_VERSION
            )
        }

        domain_data.update(self.as_dict())

        if domain_data.get(KEY_RESPONSES, {}):
            domain_data[KEY_RESPONSES] = self.get_responses_with_multilines(
                domain_data[KEY_RESPONSES]
            )

        return dump_obj_as_yaml_to_string(domain_data, should_preserve_key_order=True)

    def intent_config(self, intent_name: Text) -> Dict[Text, Any]:
        """Return the configuration for an intent."""
        return self.intent_properties.get(intent_name, {})

    @rasa.shared.utils.common.lazy_property
    def intents(self) -> List[Text]:
        """Returns sorted list of intents."""
        return sorted(self.intent_properties.keys())

    @rasa.shared.utils.common.lazy_property
    def entities(self) -> List[Text]:
        """Returns sorted list of entities."""
        return sorted(self.entity_properties.entities)

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
        """Gets the symmetric difference between two sets.

        One set represents domain elements and the other one is a set of training
        data elements.

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
            intent_properties: Dict[Text, Dict[Text, Union[bool, List]]],
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
            mappings: Optional[List[Tuple[Text, Text]]] = None,
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
            duplicates: List[Tuple[List[Text], Text]],
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

    def check_missing_responses(self) -> None:
        """Warn user of utterance names which have no specified response."""
        missing_responses = self.utterances_for_response - set(self.responses)

        for response in missing_responses:
            structlogger.warning(
                "domain.check_missing_response",
                response=response,
                event_info=(
                    f"Action '{response}' is listed as a "
                    f"response action in the domain file, but there is "
                    f"no matching response defined. Please check your domain."
                ),
                docs=DOCS_URL_RESPONSES,
            )

    def is_empty(self) -> bool:
        """Check whether the domain is empty."""
        return self.as_dict() == Domain.empty().as_dict()

    @staticmethod
    def is_domain_file(filename: Union[Text, Path]) -> bool:
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
            content = read_yaml_file(filename)
        except (RasaException, YamlSyntaxException):
            structlogger.warning(
                "domain.cannot_load_domain_file",
                file=filename,
                event_info=(
                    f"The file {filename} could not be loaded as domain file. "
                    f"You can use https://yamlchecker.com/ to validate "
                    f"the YAML syntax of your file."
                ),
            )
            return False

        return any(key in content for key in ALL_DOMAIN_KEYS)

    def required_slots_for_form(self, form_name: Text) -> List[Text]:
        """Retrieve the list of required slot names for a form defined in the domain.

        Args:
            form_name: The name of the form.

        Returns:
            The list of slot names or an empty list if no form was found.
        """
        form = self.forms.get(form_name)
        if form:
            return form[REQUIRED_SLOTS_KEY]

        return []

    def count_slot_mapping_statistics(self) -> Tuple[int, int, int]:
        """Counts the total number of slot mappings and custom slot mappings.

        Returns:
            A triple of integers where the first entry is the total number of mappings,
            the second entry is the total number of custom mappings, and the third entry
            is the total number of mappings which have conditions attached.
        """
        total_mappings = 0
        custom_mappings = 0
        conditional_mappings = 0

        for slot in self.slots:
            total_mappings += len(slot.mappings)
            for mapping in slot.mappings:
                if mapping[MAPPING_TYPE] == str(SlotMappingType.CUSTOM):
                    custom_mappings += 1

                if MAPPING_CONDITIONS in mapping:
                    conditional_mappings += 1

        return (total_mappings, custom_mappings, conditional_mappings)

    def does_custom_action_explicitly_need_domain(self, action_name: Text) -> bool:
        """Assert if action has explicitly stated that it needs domain.

        Args:
            action_name: Name of the action to be checked

        Returns:
            True if action has explicitly stated that it needs domain.
            Otherwise, it returns false.
        """
        return action_name in self._actions_which_explicitly_need_domain

    def __repr__(self) -> Text:
        """Returns text representation of object."""
        return (
            f"{self.__class__.__name__}: {len(self.action_names_or_texts)} actions, "
            f"{len(self.intent_properties)} intents, {len(self.responses)} responses, "
            f"{len(self.slots)} slots, "
            f"{len(self.entities)} entities, {len(self.form_names)} forms"
        )

    @staticmethod
    def _collect_action_names(
        actions: List[Union[Text, Dict[Text, Any]]],
    ) -> List[Text]:
        action_names: List[Text] = []

        for action in actions:
            if isinstance(action, dict):
                action_names.extend(list(action.keys()))
            elif isinstance(action, str):
                action_names += [action]

        return action_names

    @staticmethod
    def _collect_actions_which_explicitly_need_domain(
        actions: List[Union[Text, Dict[Text, Any]]],
    ) -> List[Text]:
        action_names: List[Text] = []

        for action in actions:
            if isinstance(action, dict):
                for action_name, action_config in action.items():
                    should_send_domain = action_config.get(
                        ACTION_SHOULD_SEND_DOMAIN, False
                    )
                    if should_send_domain:
                        action_names += [action_name]

            elif action.startswith("validate_"):
                action_names += [action]

        return action_names


def warn_about_duplicates_found_during_domain_merging(
    duplicates: Dict[Text, List[Text]],
) -> None:
    """Emits a warning about found duplicates while loading multiple domain paths."""
    domain_keys = [
        KEY_INTENTS,
        KEY_FORMS,
        KEY_ACTIONS,
        KEY_E2E_ACTIONS,
        KEY_RESPONSES,
        KEY_SLOTS,
        KEY_ENTITIES,
    ]

    # Build the message if there are duplicates
    message = []
    for key in domain_keys:
        duplicates_per_key = duplicates.get(key)
        if duplicates_per_key:
            message.append(
                f"The following duplicated {key} have been found "
                f"across multiple domain files: "
                f"{', '.join(duplicates_per_key)}"
            )

    # send the warning with the constructed message
    if message:
        full_message = " \n".join(message)
        structlogger.warning(
            "domain.duplicates_found", event_info=full_message, docs=DOCS_URL_DOMAINS
        )

    return None


def _validate_forms(forms: Union[Dict, List]) -> None:
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
                f"the keyword `{REQUIRED_SLOTS_KEY}` is required. "
                f"Please see {DOCS_URL_FORMS} for more information."
            )
