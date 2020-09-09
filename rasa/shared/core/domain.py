import copy
import collections
import json
from typing import Any, Dict, List, NamedTuple, Optional, Set, Text, Tuple, Union

import rasa.shared.core.constants as core_constants
import rasa.shared.constants as constants
import rasa.shared.core.slots as slots
import rasa.shared.utils.common as common_utils
import rasa.shared.utils.validation as validation_utils
import rasa.shared.utils.io as io_utils

CARRY_OVER_SLOTS_KEY = "carry_over_slots_to_new_session"
SESSION_EXPIRATION_TIME_KEY = "session_expiration_time"
SESSION_CONFIG_KEY = "session_config"
USED_ENTITIES_KEY = "used_entities"
USE_ENTITIES_KEY = "use_entities"
IGNORE_ENTITIES_KEY = "ignore_entities"

KEY_SLOTS = "slots"
KEY_INTENTS = "intents"
KEY_ENTITIES = "entities"
KEY_RESPONSES = "responses"
KEY_ACTIONS = "actions"
KEY_FORMS = "forms"
KEY_E2E_ACTIONS = "e2e_actions"


class SessionConfig(NamedTuple):
    session_expiration_time: float  # in minutes
    carry_over_slots: bool

    @staticmethod
    def default() -> "SessionConfig":
        # TODO: 2.0, reconsider how to apply sessions to old projects
        return SessionConfig(
            constants.DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES,
            constants.DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION,
        )

    def are_sessions_enabled(self) -> bool:
        return self.session_expiration_time > 0


class InvalidDomain(Exception):
    """Exception that can be raised when domain is not valid."""

    def __init__(self, message) -> None:
        self.message = message

    def __str__(self):
        # return message in error colours
        return io_utils.wrap_with_color(self.message, color=io_utils.bcolors.FAIL)


class BaseDomain:
    """The domain specifies the universe in which the bot's policy acts.

    A Domain subclass provides the actions the bot can take, the intents
    and entities it can recognise."""

    def __init__(
        self,
        intents: Union[Set[Text], List[Union[Text, Dict[Text, Any]]]],
        entities: List[Text],
        slots: List[slots.Slot],
        templates: Dict[Text, List[Dict[Text, Any]]],
        action_names: List[Text],
        forms: List[Union[Text, Dict]],
        action_texts: Optional[List[Text]] = None,
        store_entities_as_slots: bool = True,
        session_config: SessionConfig = SessionConfig.default(),
    ) -> None:
        self.intent_properties = self.collect_intent_properties(intents, entities)
        self.entities = entities

        # Forms used to be a list of form names. Now they can also contain
        # `SlotMapping`s
        if not forms or (forms and isinstance(forms[0], str)):
            self.form_names = forms
            self.forms: List[Dict] = [{form_name: {}} for form_name in forms]
        elif isinstance(forms[0], dict):
            self.forms: List[Dict] = forms
            self.form_names = [list(f.keys())[0] for f in forms]

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
                + self.form_names
                + self.action_texts
        )

        self.store_entities_as_slots = store_entities_as_slots
        self._check_domain_sanity()

    @classmethod
    def empty(cls) -> "BaseDomain":
        return cls([], [], [], {}, [], [])

    @classmethod
    def from_yaml(cls, yaml: Text, original_filename: Text = "") -> "BaseDomain":

        try:
            validation_utils.validate_yaml_schema(yaml, constants.DOMAIN_SCHEMA_FILE)
        except validation_utils.InvalidYamlFileError as e:
            raise InvalidDomain(str(e))

        data = io_utils.read_yaml(yaml)
        if not validation_utils.validate_training_data_format_version(data, original_filename):
            return BaseDomain.empty()

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict) -> "BaseDomain":
        utter_templates = data.get(KEY_RESPONSES, {})
        slots = cls._collect_slots(data.get(KEY_SLOTS, {}))
        additional_arguments = data.get("config", {})
        session_config = cls._get_session_config(data.get(SESSION_CONFIG_KEY, {}))
        intents = data.get(KEY_INTENTS, {})

        return cls(
            intents,
            data.get(KEY_ENTITIES, []),
            slots,
            utter_templates,
            data.get(KEY_ACTIONS, []),
            data.get(KEY_FORMS, []),
            data.get(KEY_E2E_ACTIONS, []),
            session_config=session_config,
            **additional_arguments,
        )

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

        for slot in domain_data[KEY_SLOTS].values():  # pytype: disable=attribute-error
            if slot["initial_value"] is None:
                del slot["initial_value"]
            if slot["auto_fill"]:
                del slot["auto_fill"]
            if slot["type"].startswith("rasa.core.slots"):
                slot["type"] = slots.Slot.resolve_by_type(slot["type"]).type_name

        if domain_data["config"]["store_entities_as_slots"]:
            del domain_data["config"]["store_entities_as_slots"]

        # clean empty keys
        return {
            k: v
            for k, v in domain_data.items()
            if v != {} and v != [] and v is not None
        }

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
        of type `UnfeaturizedSlot` from domain warnings.
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

    def is_empty(self) -> bool:
        """Check whether the domain is empty."""

        return self.as_dict() == BaseDomain.empty().as_dict()

    def as_dict(self) -> Dict[Text, Any]:
        return {
            "config": {"store_entities_as_slots": self.store_entities_as_slots},
            SESSION_CONFIG_KEY: {
                SESSION_EXPIRATION_TIME_KEY: self.session_config.session_expiration_time,
                CARRY_OVER_SLOTS_KEY: self.session_config.carry_over_slots,
            },
            KEY_INTENTS: self._transform_intents_for_file(),
            KEY_ENTITIES: self.entities,
            KEY_SLOTS: self._slot_definitions(),
            KEY_RESPONSES: self.templates,
            KEY_ACTIONS: self._custom_actions,  # class names of the actions
            KEY_FORMS: self.forms,
            KEY_E2E_ACTIONS: self.action_texts,
        }

    @common_utils.lazy_property
    def intents(self):
        return sorted(self.intent_properties.keys())

    @common_utils.lazy_property
    def user_actions_and_forms(self):
        """Returns combination of user actions and forms."""

        return self.user_actions + self.form_names

    @property
    def _slots_for_domain_warnings(self) -> List[Text]:
        """Fetch names of slots that are used in domain warnings.

        Excludes slots of type `UnfeaturizedSlot`.
        """

        return [s.name for s in self.slots if not isinstance(s, slots.UnfeaturizedSlot)]

    @property
    def _actions_for_domain_warnings(self) -> List[Text]:
        """Fetch names of actions that are used in domain warnings.

        Includes user and form actions, but excludes those that are default actions.
        """
        return [
            a for a in self.user_actions_and_forms if a not in core_constants.DEFAULT_ACTIONS
        ]

    @staticmethod
    def _collect_slots(slot_dict: Dict[Text, Any]) -> List[slots.Slot]:
        # it is super important to sort the slots here!!!
        # otherwise state ordering is not consistent
        slots_list = []
        # make a copy to not alter the input dictionary
        slot_dict = copy.deepcopy(slot_dict)
        for slot_name in sorted(slot_dict):
            slot_class = slots.Slot.resolve_by_type(slot_dict[slot_name].get("type"))
            if "type" in slot_dict[slot_name]:
                del slot_dict[slot_name]["type"]
            slot = slot_class(slot_name, **slot_dict[slot_name])
            slots_list.append(slot)
        return slots_list

    @staticmethod
    def _get_session_config(session_config: Dict) -> SessionConfig:
        session_expiration_time_min = session_config.get(SESSION_EXPIRATION_TIME_KEY)

        # TODO: 2.0 reconsider how to apply sessions to old projects and legacy trackers
        if session_expiration_time_min is None:
            session_expiration_time_min = constants.DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES

        carry_over_slots = session_config.get(
            CARRY_OVER_SLOTS_KEY, constants.DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION
        )

        return SessionConfig(session_expiration_time_min, carry_over_slots)

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

    @classmethod
    def collect_intent_properties(
            cls, intents: List[Union[Text, Dict[Text, Any]]], entities: List[Text]
    ) -> Dict[Text, Dict[Text, Union[bool, List]]]:
        """Get intent properties for a domain from what is provided by a domain file.
        Args:
            intents: The intents as provided by a domain file.
            entities: All entities as provided by a domain file.
        Returns:
            The intent properties to be stored in the domain.
        """
        # make a copy to not alter the input argument
        intents = copy.deepcopy(intents)
        intent_properties = {}
        duplicates = set()

        for intent in intents:
            intent_name, properties = cls._intent_properties(intent, entities)

            if intent_name in intent_properties.keys():
                duplicates.add(intent_name)

            intent_properties.update(properties)

        if duplicates:
            raise InvalidDomain(
                f"Intents are not unique! Found multiple intents with name(s) {sorted(duplicates)}. "
                f"Either rename or remove the duplicate ones."
            )

        cls._add_default_intents(intent_properties, entities)

        return intent_properties

    @classmethod
    def _intent_properties(
            cls, intent: Union[Text, Dict[Text, Any]], entities: List[Text]
    ) -> Tuple[Text, Dict[Text, Any]]:
        if not isinstance(intent, dict):
            intent_name = intent
            intent = {intent_name: {USE_ENTITIES_KEY: True, IGNORE_ENTITIES_KEY: []}}
        else:
            intent_name = list(intent.keys())[0]

        return (
            intent_name,
            cls._transform_intent_properties_for_internal_use(intent, entities),
        )

    @classmethod
    def _add_default_intents(
            cls,
            intent_properties: Dict[Text, Dict[Text, Union[bool, List]]],
            entities: List[Text],
    ) -> None:
        for intent_name in core_constants.DEFAULT_INTENTS:
            if intent_name not in intent_properties:
                _, properties = cls._intent_properties(intent_name, entities)
                intent_properties.update(properties)

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
            io_utils.raise_warning(
                f"Entities: '{ambiguous_entities}' are explicitly included and"
                f" excluded for intent '{name}'."
                f"Excluding takes precedence in this case. "
                f"Please resolve that ambiguity.",
                docs=f"{constants.DOCS_URL_DOMAINS}#ignoring-entities-for-certain-intents",
            )

        properties[USED_ENTITIES_KEY] = used_entities
        del properties[USE_ENTITIES_KEY]
        del properties[IGNORE_ENTITIES_KEY]

        return intent

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
        unique_user_actions = [a for a in user_actions if a not in core_constants.DEFAULT_ACTIONS]
        return core_constants.DEFAULT_ACTIONS + unique_user_actions

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
            if intent_name in core_constants.DEFAULT_INTENTS:
                # Default intents should be not dumped with the domain
                continue
            use_entities = set(intent_props[USED_ENTITIES_KEY])
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

    def _slot_definitions(self) -> Dict[Any, Dict[str, Any]]:
        return {slot.name: slot.persistence_info() for slot in self.slots}

    def __hash__(self) -> int:
        self_as_dict = self.as_dict()
        self_as_dict[KEY_INTENTS] = common_utils.sort_list_of_dicts_by_first_key(
            self_as_dict[KEY_INTENTS]
        )
        self_as_dict[KEY_ACTIONS] = self.action_names
        self_as_string = json.dumps(self_as_dict, sort_keys=True)
        text_hash = io_utils.get_text_hash(self_as_string)

        return int(text_hash, 16)
