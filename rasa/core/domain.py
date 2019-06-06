import collections
import json
import logging
import os
import typing
from typing import Any, Dict, List, Optional, Text, Tuple, Union, Set

import pkg_resources
from pykwalify.errors import SchemaError
from ruamel.yaml import YAMLError
from ruamel.yaml.constructor import DuplicateKeyError

import rasa.utils.io
from rasa import data
from rasa.cli.utils import print_warning, bcolors
from rasa.core import utils
from rasa.core.actions import Action, action
from rasa.core.constants import REQUESTED_SLOT
from rasa.core.events import SlotSet
from rasa.core.slots import Slot, UnfeaturizedSlot
from rasa.skill import SkillSelector
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)

PREV_PREFIX = "prev_"
ACTIVE_FORM_PREFIX = "active_form_"

if typing.TYPE_CHECKING:
    from rasa.core.trackers import DialogueStateTracker


class InvalidDomain(Exception):
    """Exception that can be raised when domain is not valid."""

    def __init__(self, message):
        self.message = message

    def __str__(self):
        # return message in error colours
        return bcolors.FAIL + self.message + bcolors.ENDC


class Domain(object):
    """The domain specifies the universe in which the bot's policy acts.

    A Domain subclass provides the actions the bot can take, the intents
    and entities it can recognise"""

    @classmethod
    def empty(cls) -> "Domain":
        return cls({}, [], [], {}, [], [])

    @classmethod
    def load(
        cls,
        paths: Union[List[Text], Text],
        skill_imports: Optional[SkillSelector] = None,
    ) -> "Domain":
        skill_imports = skill_imports or SkillSelector.all_skills()

        if not skill_imports.no_skills_selected():
            paths = skill_imports.training_paths()

        if not paths:
            raise InvalidDomain(
                "No domain file was specified. Please specify a path "
                "to a valid domain file."
            )
        elif not isinstance(paths, list) and not isinstance(paths, set):
            paths = [paths]

        domain = Domain.empty()
        for path in paths:
            other = cls.from_path(path, skill_imports)
            domain = domain.merge(other)

        return domain

    @classmethod
    def from_path(cls, path: Text, skill_imports: SkillSelector) -> "Domain":
        path = os.path.abspath(path)

        # If skills were imported search the whole directory tree for domain files
        if os.path.isfile(path) and not skill_imports.no_skills_selected():
            path = os.path.dirname(path)

        if os.path.isfile(path):
            domain = cls.from_file(path)
        elif os.path.isdir(path):
            domain = cls.from_directory(path, skill_imports)
        else:
            raise Exception(
                "Failed to load domain specification from '{}'. "
                "File not found!".format(os.path.abspath(path))
            )

        return domain

    @classmethod
    def from_file(cls, path: Text) -> "Domain":
        return cls.from_yaml(rasa.utils.io.read_file(path))

    @classmethod
    def from_yaml(cls, yaml: Text) -> "Domain":
        cls.validate_domain_yaml(yaml)
        data = rasa.utils.io.read_yaml(yaml)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict) -> "Domain":
        utter_templates = cls.collect_templates(data.get("templates", {}))
        slots = cls.collect_slots(data.get("slots", {}))
        additional_arguments = data.get("config", {})
        intent_properties = cls.collect_intent_properties(data.get("intents", {}))
        return cls(
            intent_properties,
            data.get("entities", []),
            slots,
            utter_templates,
            data.get("actions", []),
            data.get("forms", []),
            **additional_arguments
        )

    @classmethod
    def from_directory(
        cls, path: Text, skill_imports: Optional[SkillSelector] = None
    ) -> "Domain":
        """Loads and merges multiple domain files recursively from a directory tree."""

        domain = Domain.empty()
        skill_imports = skill_imports or SkillSelector.all_skills()

        for root, _, files in os.walk(path):
            if not skill_imports.is_imported(root):
                continue

            for file in files:
                full_path = os.path.join(root, file)
                if data.is_domain_file(full_path):
                    other = Domain.from_file(full_path)
                    domain = other.merge(domain)

        return domain

    def merge(self, domain: "Domain", override: bool = False) -> "Domain":
        """Merge this domain with another one, combining their attributes.

        List attributes like ``intents`` and ``actions`` will be deduped
        and merged. Single attributes will be taken from ``self`` unless
        override is `True`, in which case they are taken from ``domain``."""

        domain_dict = domain.as_dict()
        combined = self.as_dict()

        def merge_dicts(d1, d2, override_existing_values=False):
            if override_existing_values:
                a, b = d1.copy(), d2.copy()
            else:
                a, b = d2.copy(), d1.copy()
            a.update(b)
            return a

        def merge_lists(l1, l2):
            return sorted(list(set(l1 + l2)))

        if override:
            for key, val in domain_dict["config"].items():
                combined["config"][key] = val

        # intents is list of dicts
        intents_1 = {list(i.keys())[0]: i for i in combined["intents"]}
        intents_2 = {list(i.keys())[0]: i for i in domain_dict["intents"]}
        merged_intents = merge_dicts(intents_1, intents_2, override)
        combined["intents"] = list(merged_intents.values())

        # remove existing forms from new actions
        for form in combined["forms"]:
            if form in domain_dict["actions"]:
                domain_dict["actions"].remove(form)

        for key in ["entities", "actions", "forms"]:
            combined[key] = merge_lists(combined[key], domain_dict[key])

        for key in ["templates", "slots"]:
            combined[key] = merge_dicts(combined[key], domain_dict[key], override)

        return self.__class__.from_dict(combined)

    @classmethod
    def validate_domain_yaml(cls, yaml):
        """Validate domain yaml."""
        from pykwalify.core import Core

        log = logging.getLogger("pykwalify")
        log.setLevel(logging.WARN)

        try:
            schema_file = pkg_resources.resource_filename(
                __name__, "schemas/domain.yml"
            )
            source_data = rasa.utils.io.read_yaml(yaml)
        except YAMLError:
            raise InvalidDomain(
                "The provided domain file is invalid. You can use "
                "http://www.yamllint.com/ to validate the yaml syntax "
                "of your domain file."
            )
        except DuplicateKeyError as e:
            raise InvalidDomain(
                "The provided domain file contains a duplicated key: {}".format(str(e))
            )

        try:
            c = Core(source_data=source_data, schema_files=[schema_file])
            c.validate(raise_exception=True)
        except SchemaError:
            raise InvalidDomain(
                "Failed to validate your domain yaml. "
                "Please make sure the file is correct; to do so, "
                "take a look at the errors logged during "
                "validation previous to this exception. "
                "You can also validate your domain file's yaml "
                "syntax using http://www.yamllint.com/."
            )

    @staticmethod
    def collect_slots(slot_dict):
        # it is super important to sort the slots here!!!
        # otherwise state ordering is not consistent
        slots = []
        for slot_name in sorted(slot_dict):
            slot_class = Slot.resolve_by_type(slot_dict[slot_name].get("type"))
            if "type" in slot_dict[slot_name]:
                del slot_dict[slot_name]["type"]
            slot = slot_class(slot_name, **slot_dict[slot_name])
            slots.append(slot)
        return slots

    @staticmethod
    def collect_intent_properties(intent_list):
        intent_properties = {}
        for intent in intent_list:
            if isinstance(intent, dict):
                name = list(intent.keys())[0]
                for properties in intent.values():
                    if "use_entities" not in properties:
                        properties["use_entities"] = True
            else:
                name = intent
                intent = {intent: {"use_entities": True}}

            if name in intent_properties.keys():
                raise InvalidDomain(
                    "Intents are not unique! Found two intents with name '{}'. "
                    "Either rename or remove one of them.".format(name)
                )

            intent_properties.update(intent)
        return intent_properties

    @staticmethod
    def collect_templates(
        yml_templates: Dict[Text, List[Any]]
    ) -> Dict[Text, List[Dict[Text, Any]]]:
        """Go through the templates and make sure they are all in dict format
        """
        templates = {}
        for template_key, template_variations in yml_templates.items():
            validated_variations = []
            if template_variations is None:
                raise InvalidDomain(
                    "Utterance '{}' does not have any defined templates.".format(
                        template_key
                    )
                )

            for t in template_variations:

                # templates should be a dict with options
                if isinstance(t, str):
                    logger.warning(
                        "Deprecated: Templates should not be strings anymore. "
                        "Utterance template '{}' should contain either '- text: ' or "
                        "'- custom: ' attribute to be a proper template.".format(
                            template_key
                        )
                    )
                    validated_variations.append({"text": t})
                elif "text" not in t and "custom" not in t:
                    raise InvalidDomain(
                        "Utter template '{}' needs to contain either "
                        "'- text: ' or '- custom: ' attribute to be a proper "
                        "template.".format(template_key)
                    )
                else:
                    validated_variations.append(t)

            templates[template_key] = validated_variations
        return templates

    def __init__(
        self,
        intent_properties: Dict[Text, Any],
        entities: List[Text],
        slots: List[Slot],
        templates: Dict[Text, Any],
        action_names: List[Text],
        form_names: List[Text],
        store_entities_as_slots: bool = True,
    ) -> None:

        self.intent_properties = intent_properties
        self.entities = entities
        self.form_names = form_names
        self.slots = slots
        self.templates = templates

        # only includes custom actions and utterance actions
        self.user_actions = action_names
        # includes all actions (custom, utterance, default actions and forms)
        self.action_names = (
            action.combine_user_with_default_actions(action_names) + form_names
        )
        self.store_entities_as_slots = store_entities_as_slots

        self._check_domain_sanity()

    def __hash__(self) -> int:
        from rasa.utils.common import sort_list_of_dicts_by_first_key

        self_as_dict = self.as_dict()
        self_as_dict["intents"] = sort_list_of_dicts_by_first_key(
            self_as_dict["intents"]
        )
        self_as_string = json.dumps(self_as_dict, sort_keys=True)
        text_hash = utils.get_text_hash(self_as_string)

        return int(text_hash, 16)

    @utils.lazyproperty
    def user_actions_and_forms(self):
        """Returns combination of user actions and forms"""

        return self.user_actions + self.form_names

    @utils.lazyproperty
    def num_actions(self):
        """Returns the number of available actions."""

        # noinspection PyTypeChecker
        return len(self.action_names)

    @utils.lazyproperty
    def num_states(self):
        """Number of used input states for the action prediction."""

        return len(self.input_states)

    def add_requested_slot(self):
        if self.form_names and REQUESTED_SLOT not in [s.name for s in self.slots]:
            self.slots.append(UnfeaturizedSlot(REQUESTED_SLOT))

    def action_for_name(
        self, action_name: Text, action_endpoint: Optional[EndpointConfig]
    ) -> Optional[Action]:
        """Looks up which action corresponds to this action name."""

        if action_name not in self.action_names:
            self._raise_action_not_found_exception(action_name)

        return action.action_from_name(
            action_name, action_endpoint, self.user_actions_and_forms
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

    def actions(self, action_endpoint):
        return [
            self.action_for_name(name, action_endpoint) for name in self.action_names
        ]

    def index_for_action(self, action_name: Text) -> Optional[int]:
        """Looks up which action index corresponds to this action name"""

        try:
            return self.action_names.index(action_name)
        except ValueError:
            self._raise_action_not_found_exception(action_name)

    def _raise_action_not_found_exception(self, action_name):
        action_names = "\n".join(["\t - {}".format(a) for a in self.action_names])
        raise NameError(
            "Cannot access action '{}', "
            "as that name is not a registered "
            "action for this domain. "
            "Available actions are: \n{}"
            "".format(action_name, action_names)
        )

    def random_template_for(self, utter_action):
        import numpy as np

        if utter_action in self.templates:
            return np.random.choice(self.templates[utter_action])
        else:
            return None

    # noinspection PyTypeChecker
    @utils.lazyproperty
    def slot_states(self):
        # type: () -> List[Text]
        """Returns all available slot state strings."""

        return [
            "slot_{}_{}".format(s.name, i)
            for s in self.slots
            for i in range(0, s.feature_dimensionality())
        ]

    # noinspection PyTypeChecker
    @utils.lazyproperty
    def prev_action_states(self) -> List[Text]:
        """Returns all available previous action state strings."""

        return [PREV_PREFIX + a for a in self.action_names]

    # noinspection PyTypeChecker
    @utils.lazyproperty
    def intent_states(self) -> List[Text]:
        """Returns all available previous action state strings."""

        return ["intent_{0}".format(i) for i in self.intents]

    # noinspection PyTypeChecker
    @utils.lazyproperty
    def entity_states(self) -> List[Text]:
        """Returns all available previous action state strings."""

        return ["entity_{0}".format(e) for e in self.entities]

    # noinspection PyTypeChecker
    @utils.lazyproperty
    def form_states(self) -> List[Text]:
        return ["active_form_{0}".format(f) for f in self.form_names]

    def index_of_state(self, state_name: Text) -> Optional[int]:
        """Provides the index of a state."""

        return self.input_state_map.get(state_name)

    @utils.lazyproperty
    def input_state_map(self) -> Dict[Text, int]:
        """Provides a mapping from state names to indices."""
        return {f: i for i, f in enumerate(self.input_states)}

    @utils.lazyproperty
    def input_states(self) -> List[Text]:
        """Returns all available states."""

        return (
            self.intent_states
            + self.entity_states
            + self.slot_states
            + self.prev_action_states
            + self.form_states
        )

    def get_parsing_states(self, tracker: "DialogueStateTracker") -> Dict[Text, float]:
        state_dict = {}

        # Set all found entities with the state value 1.0, unless they should
        # be ignored for the current intent
        for entity in tracker.latest_message.entities:
            intent_name = tracker.latest_message.intent.get("name")
            intent_config = self.intent_config(intent_name)
            should_use_entity = intent_config.get("use_entities", True)
            if should_use_entity:
                if "entity" in entity:
                    key = "entity_{0}".format(entity["entity"])
                    state_dict[key] = 1.0

        # Set all set slots with the featurization of the stored value
        for key, slot in tracker.slots.items():
            if slot is not None:
                for i, slot_value in enumerate(slot.as_feature()):
                    if slot_value != 0:
                        slot_id = "slot_{}_{}".format(key, i)
                        state_dict[slot_id] = slot_value

        latest_message = tracker.latest_message

        if "intent_ranking" in latest_message.parse_data:
            for intent in latest_message.parse_data["intent_ranking"]:
                if intent.get("name"):
                    intent_id = "intent_{}".format(intent["name"])
                    state_dict[intent_id] = intent["confidence"]

        elif latest_message.intent.get("name"):
            intent_id = "intent_{}".format(latest_message.intent["name"])
            state_dict[intent_id] = latest_message.intent.get("confidence", 1.0)

        return state_dict

    def get_prev_action_states(
        self, tracker: "DialogueStateTracker"
    ) -> Dict[Text, float]:
        """Turns the previous taken action into a state name."""

        latest_action = tracker.latest_action_name
        if latest_action:
            prev_action_name = PREV_PREFIX + latest_action
            if prev_action_name in self.input_state_map:
                return {prev_action_name: 1.0}
            else:
                logger.warning(
                    "Failed to use action '{}' in history. "
                    "Please make sure all actions are listed in the "
                    "domains action list. If you recently removed an "
                    "action, don't worry about this warning. It "
                    "should stop appearing after a while. "
                    "".format(latest_action)
                )
                return {}
        else:
            return {}

    @staticmethod
    def get_active_form(tracker: "DialogueStateTracker") -> Dict[Text, float]:
        """Turns tracker's active form into a state name."""
        form = tracker.active_form.get("name")
        if form is not None:
            return {ACTIVE_FORM_PREFIX + form: 1.0}
        else:
            return {}

    def get_active_states(self, tracker: "DialogueStateTracker") -> Dict[Text, float]:
        """Return a bag of active states from the tracker state"""
        state_dict = self.get_parsing_states(tracker)
        state_dict.update(self.get_prev_action_states(tracker))
        state_dict.update(self.get_active_form(tracker))
        return state_dict

    def states_for_tracker_history(
        self, tracker: "DialogueStateTracker"
    ) -> List[Dict[Text, float]]:
        """Array of states for each state of the trackers history."""
        return [
            self.get_active_states(tr) for tr in tracker.generate_all_prior_trackers()
        ]

    def slots_for_entities(self, entities):
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
        """Persists the domain specification to storage."""

        domain_spec_path = os.path.join(model_path, "domain.json")
        rasa.utils.io.create_directory_for_file(domain_spec_path)

        metadata = {"states": self.input_states}
        utils.dump_obj_as_json_to_file(domain_spec_path, metadata)

    @classmethod
    def load_specification(cls, path: Text) -> Dict[Text, Any]:
        """Load a domains specification from a dumped model directory."""

        metadata_path = os.path.join(path, "domain.json")
        specification = json.loads(rasa.utils.io.read_file(metadata_path))
        return specification

    def compare_with_specification(self, path: Text) -> bool:
        """Compares the domain spec of the current and the loaded domain.

        Throws exception if the loaded domain specification is different
        to the current domain are different."""

        loaded_domain_spec = self.load_specification(path)
        states = loaded_domain_spec["states"]

        if states != self.input_states:
            missing = ",".join(set(states) - set(self.input_states))
            additional = ",".join(set(self.input_states) - set(states))
            raise InvalidDomain(
                "Domain specification has changed. "
                "You MUST retrain the policy. "
                + "Detected mismatch in domain specification. "
                + "The following states have been \n"
                "\t - removed: {} \n"
                "\t - added:   {} ".format(missing, additional)
            )
        else:
            return True

    def _slot_definitions(self):
        return {slot.name: slot.persistence_info() for slot in self.slots}

    def as_dict(self) -> Dict[Text, Any]:
        additional_config = {"store_entities_as_slots": self.store_entities_as_slots}

        return {
            "config": additional_config,
            "intents": [{k: v} for k, v in self.intent_properties.items()],
            "entities": self.entities,
            "slots": self._slot_definitions(),
            "templates": self.templates,
            "actions": self.user_actions,  # class names of the actions
            "forms": self.form_names,
        }

    def persist(self, filename: Text) -> None:
        """Write domain to a file."""

        domain_data = self.as_dict()
        utils.dump_obj_as_yaml_to_file(filename, domain_data)

    def cleaned_domain(self) -> Dict[Text, Any]:
        """Fetch cleaned domain, replacing redundant keys with default values."""

        domain_data = self.as_dict()
        for idx, intent_info in enumerate(domain_data["intents"]):
            for name, intent in intent_info.items():
                if intent.get("use_entities"):
                    domain_data["intents"][idx] = name

        for slot in domain_data["slots"].values():
            if slot["initial_value"] is None:
                del slot["initial_value"]
            if slot["auto_fill"]:
                del slot["auto_fill"]
            if slot["type"].startswith("rasa.core.slots"):
                slot["type"] = Slot.resolve_by_type(slot["type"]).type_name

        if domain_data["config"]["store_entities_as_slots"]:
            del domain_data["config"]["store_entities_as_slots"]

        # clean empty keys
        return {
            k: v
            for k, v in domain_data.items()
            if v != {} and v != [] and v is not None
        }

    def persist_clean(self, filename: Text) -> None:
        """Write cleaned domain to a file."""

        cleaned_domain_data = self.cleaned_domain()
        utils.dump_obj_as_yaml_to_file(filename, cleaned_domain_data)

    def as_yaml(self, clean_before_dump=False):
        if clean_before_dump:
            domain_data = self.cleaned_domain()
        else:
            domain_data = self.as_dict()

        return utils.dump_obj_as_yaml_to_string(domain_data)

    def intent_config(self, intent_name: Text) -> Dict[Text, Any]:
        """Return the configuration for an intent."""
        return self.intent_properties.get(intent_name, {})

    @utils.lazyproperty
    def intents(self):
        return sorted(self.intent_properties.keys())

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

    def domain_warnings(
        self,
        intents: Optional[Union[List[Text], Set[Text]]] = None,
        entities: Optional[Union[List[Text], Set[Text]]] = None,
        actions: Optional[Union[List[Text], Set[Text]]] = None,
        slots: Optional[Union[List[Text], Set[Text]]] = None,
    ) -> Dict[Text, Dict[Text, Set[Text]]]:
        """Generate domain warnings from intents, entities, actions and slots.

        Returns a dictionary with entries for `intent_warnings`,
        `entity_warnings`, `action_warnings` and `slot_warnings`.
        """

        intent_warnings = self._get_symmetric_difference(self.intents, intents)
        entity_warnings = self._get_symmetric_difference(self.entities, entities)
        action_warnings = self._get_symmetric_difference(self.user_actions, actions)
        slot_warnings = self._get_symmetric_difference(
            [s.name for s in self.slots], slots
        )

        return {
            "intent_warnings": intent_warnings,
            "entity_warnings": entity_warnings,
            "action_warnings": action_warnings,
            "slot_warnings": slot_warnings,
        }

    def _check_domain_sanity(self):
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

        def check_mappings(intent_properties):
            """Check whether intent-action mappings use proper action names."""

            incorrect = list()
            for intent, properties in intent_properties.items():
                if "triggers" in properties:
                    if properties.get("triggers") not in self.action_names:
                        incorrect.append((intent, properties["triggers"]))
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
                        "Duplicate {0} in domain. "
                        "These {0} occur more than once in "
                        "the domain: '{1}'".format(name, "', '".join(d))
                    )
            return message

        def warn_missing_templates(
            action_names: List[Text], templates: Dict[Text, Any]
        ) -> None:
            """Warn user of utterance names which have no specified template."""

            utterances = [
                act for act in action_names if act.startswith(action.UTTER_PREFIX)
            ]

            missing_templates = [t for t in utterances if t not in templates.keys()]

            if missing_templates:
                message = ""
                for template in missing_templates:
                    message += (
                        "\nUtterance '{}' is listed as an "
                        "action in the domain file, but there is "
                        "no matching utterance template. Please "
                        "check your domain."
                    ).format(template)
                print_warning(message)

        warn_missing_templates(self.action_names, self.templates)
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
                        (duplicate_actions, "actions"),
                        (duplicate_slots, "slots"),
                        (duplicate_entities, "entities"),
                    ],
                    incorrect_mappings,
                )
            )


class TemplateDomain(Domain):
    pass
