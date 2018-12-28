import typing

import collections
import json
import logging
import numpy as np
import os
import pkg_resources
from pykwalify.errors import SchemaError
from rasa_core import utils
from rasa_core.actions import Action, action
from rasa_core.constants import REQUESTED_SLOT
from rasa_core.slots import Slot, UnfeaturizedSlot
from rasa_core.trackers import SlotSet
from rasa_core.utils import read_file, read_yaml_string, EndpointConfig
from typing import Dict, Any, Tuple
from typing import List
from typing import Optional
from typing import Text

logger = logging.getLogger(__name__)

PREV_PREFIX = 'prev_'
ACTIVE_FORM_PREFIX = 'active_form_'

if typing.TYPE_CHECKING:
    from rasa_core.trackers import DialogueStateTracker


class InvalidDomain(Exception):
    """Exception that can be raised when domain is not valid."""
    pass


def check_domain_sanity(domain):
    """Make sure the domain is properly configured.

    Checks the settings and checks if there are duplicate actions,
    intents, slots and entities."""

    def get_duplicates(my_items):
        """Returns a list of duplicate items in my_items."""
        return [item
                for item, count in collections.Counter(my_items).items()
                if count > 1]

    def get_exception_message(
        duplicates: List[Tuple[List[Text], Text]]
    ) -> Text:
        """Return a message given a list of error locations."""

        msg = ""
        for d, name in duplicates:
            if d:
                if msg:
                    msg += "\n"
                msg += ("Duplicate {0} in domain. "
                        "These {0} occur more than once in "
                        "the domain: {1}".format(name, ", ".join(d)))
        return msg

    duplicate_actions = get_duplicates(domain.action_names[:])
    duplicate_intents = get_duplicates(domain.intents[:])
    duplicate_slots = get_duplicates([s.name for s in domain.slots])
    duplicate_entities = get_duplicates(domain.entities[:])

    if (duplicate_actions or duplicate_intents or
            duplicate_slots or duplicate_entities):
        raise InvalidDomain(get_exception_message([
            (duplicate_actions, "actions"),
            (duplicate_intents, "intents"),
            (duplicate_slots, "slots"),
            (duplicate_entities, "entities")]))


class Domain(object):
    """The domain specifies the universe in which the bot's policy acts.

    A Domain subclass provides the actions the bot can take, the intents
    and entities it can recognise"""

    @classmethod
    def load(cls, filename):
        if not os.path.isfile(filename):
            raise Exception(
                "Failed to load domain specification from '{}'. "
                "File not found!".format(os.path.abspath(filename)))
        return cls.from_yaml(read_file(filename))

    @classmethod
    def from_yaml(cls, yaml):
        cls.validate_domain_yaml(yaml)
        data = read_yaml_string(yaml)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data):
        utter_templates = cls.collect_templates(data.get("templates", {}))
        slots = cls.collect_slots(data.get("slots", {}))
        additional_arguments = data.get("config", {})
        intent_properties = cls.collect_intent_properties(data.get("intents",
                                                                   {}))
        return cls(
            intent_properties,
            data.get("entities", []),
            slots,
            utter_templates,
            data.get("actions", []),
            data.get("forms", []),
            **additional_arguments
        )

    def merge(self, domain: 'Domain', override: bool = False) -> 'Domain':
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
            return list(set(l1 + l2))

        if override:
            for key, val in domain_dict['config'].items():
                combined['config'][key] = val

        # intents is list of dicts
        intents_1 = {list(i.keys())[0]: i for i in combined["intents"]}
        intents_2 = {list(i.keys())[0]: i for i in domain_dict["intents"]}
        merged_intents = merge_dicts(intents_1, intents_2, override)
        combined['intents'] = list(merged_intents.values())

        # remove existing forms from new actions
        for form in combined['forms']:
            if form in domain_dict['actions']:
                domain_dict['actions'].remove(form)

        for key in ['entities', 'actions', 'forms']:
            combined[key] = merge_lists(combined[key],
                                        domain_dict[key])

        for key in ['templates', 'slots']:
            combined[key] = merge_dicts(combined[key],
                                        domain_dict[key],
                                        override)

        return self.__class__.from_dict(combined)

    @classmethod
    def validate_domain_yaml(cls, yaml):
        """Validate domain yaml."""
        from pykwalify.core import Core

        log = logging.getLogger('pykwalify')
        log.setLevel(logging.WARN)

        schema_file = pkg_resources.resource_filename(__name__,
                                                      "schemas/domain.yml")
        source_data = utils.read_yaml_string(yaml)
        c = Core(source_data=source_data,
                 schema_files=[schema_file])
        try:
            c.validate(raise_exception=True)
        except SchemaError:
            raise InvalidDomain("Failed to validate your domain yaml. "
                                "Make sure the file is correct, to do so"
                                "take a look at the errors logged during "
                                "validation previous to this exception. ")

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
                intent_properties.update(intent)
            else:
                intent_properties.update({intent: {'use_entities': True}})
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
            for t in template_variations:
                # templates can either directly be strings or a dict with
                # options we will always create a dict out of them
                if isinstance(t, str):
                    validated_variations.append({"text": t})
                elif "text" not in t:
                    raise InvalidDomain("Utter template '{}' needs to contain"
                                        "'- text: ' attribute to be a proper"
                                        "template".format(template_key))
                else:
                    validated_variations.append(t)
            templates[template_key] = validated_variations
        return templates

    def __init__(self,
                 intent_properties: Dict[Text, Any],
                 entities: List[Text],
                 slots: List[Slot],
                 templates: Dict[Text, Any],
                 action_names: List[Text],
                 form_names: List[Text],
                 store_entities_as_slots: bool = True,
                 restart_intent="restart"  # type: Text
                 ) -> None:

        self.intent_properties = intent_properties
        self.entities = entities
        self.form_names = form_names
        self.slots = slots
        self.templates = templates

        # only includes custom actions and utterance actions
        self.user_actions = action_names
        # includes all actions (custom, utterance, default actions and forms)
        self.action_names = action.combine_user_with_default_actions(
            action_names) + form_names
        self.store_entities_as_slots = store_entities_as_slots
        self.restart_intent = restart_intent

        action.ensure_action_name_uniqueness(self.action_names)

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
        if self.form_names and REQUESTED_SLOT not in [s.name for
                                                      s in self.slots]:
            self.slots.append(UnfeaturizedSlot(REQUESTED_SLOT))

    def action_for_name(self,
                        action_name: Text,
                        action_endpoint: Optional[EndpointConfig]
                        ) -> Optional[Action]:
        """Looks up which action corresponds to this action name."""

        if action_name not in self.action_names:
            self._raise_action_not_found_exception(action_name)

        return action.action_from_name(action_name,
                                       action_endpoint,
                                       self.user_actions_and_forms)

    def action_for_index(self,
                         index: int,
                         action_endpoint: Optional[EndpointConfig]
                         ) -> Optional[Action]:
        """Integer index corresponding to an actions index in the action list.

        This method resolves the index to the actions name."""

        if self.num_actions <= index or index < 0:
            raise IndexError("Cannot access action at index {}. "
                             "Domain has {} actions."
                             "".format(index, self.num_actions))

        return self.action_for_name(self.action_names[index],
                                    action_endpoint)

    def actions(self, action_endpoint):
        return [self.action_for_name(name, action_endpoint)
                for name in self.action_names]

    def index_for_action(self, action_name: Text) -> Optional[int]:
        """Looks up which action index corresponds to this action name"""

        try:
            return self.action_names.index(action_name)
        except ValueError:
            self._raise_action_not_found_exception(action_name)

    def _raise_action_not_found_exception(self, action_name):
        action_names = "\n".join(["\t - {}".format(a)
                                  for a in self.action_names])
        raise NameError("Cannot access action '{}', "
                        "as that name is not a registered "
                        "action for this domain. "
                        "Available actions are: \n{}"
                        "".format(action_name, action_names))

    def random_template_for(self, utter_action):
        if utter_action in self.templates:
            return np.random.choice(self.templates[utter_action])
        else:
            return None

    # noinspection PyTypeChecker
    @utils.lazyproperty
    def slot_states(self):
        # type: () -> List[Text]
        """Returns all available slot state strings."""

        return ["slot_{}_{}".format(s.name, i)
                for s in self.slots
                for i in range(0, s.feature_dimensionality())]

    # noinspection PyTypeChecker
    @utils.lazyproperty
    def prev_action_states(self):
        # type: () -> List[Text]
        """Returns all available previous action state strings."""

        return [PREV_PREFIX + a for a in self.action_names]

    # noinspection PyTypeChecker
    @utils.lazyproperty
    def intent_states(self):
        # type: () -> List[Text]
        """Returns all available previous action state strings."""

        return ["intent_{0}".format(i)
                for i in self.intents]

    # noinspection PyTypeChecker
    @utils.lazyproperty
    def entity_states(self):
        # type: () -> List[Text]
        """Returns all available previous action state strings."""

        return ["entity_{0}".format(e)
                for e in self.entities]

    # noinspection PyTypeChecker
    @utils.lazyproperty
    def form_states(self):
        # type: () -> List[Text]
        return ["active_form_{0}".format(f) for f in self.form_names]

    def index_of_state(self, state_name: Text) -> Optional[int]:
        """Provides the index of a state."""

        return self.input_state_map.get(state_name)

    @utils.lazyproperty
    def input_state_map(self):
        # type: () -> Dict[Text, int]
        """Provides a mapping from state names to indices."""
        return {f: i for i, f in enumerate(self.input_states)}

    @utils.lazyproperty
    def input_states(self):
        # type: () -> List[Text]
        """Returns all available states."""

        return \
            self.intent_states + \
            self.entity_states + \
            self.slot_states + \
            self.prev_action_states + \
            self.form_states

    def get_parsing_states(self,
                           tracker: 'DialogueStateTracker'
                           ) -> Dict[Text, float]:

        state_dict = {}

        # Set all found entities with the state value 1.0, unless they should
        # be ignored for the current intent
        for entity in tracker.latest_message.entities:
            intent_name = tracker.latest_message.intent.get("name")
            intent_config = self.intent_config(intent_name)
            should_use_entity = intent_config.get('use_entities', True)
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
            state_dict[intent_id] = latest_message.intent.get("confidence",
                                                              1.0)

        return state_dict

    def get_prev_action_states(self,
                               tracker: 'DialogueStateTracker'
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
                    "".format(latest_action))
                return {}
        else:
            return {}

    @staticmethod
    def get_active_form(tracker: 'DialogueStateTracker') -> Dict[Text, float]:
        """Turns tracker's active form into a state name."""
        form = tracker.active_form.get('name')
        if form is not None:
            return {ACTIVE_FORM_PREFIX + form: 1.0}
        else:
            return {}

    def get_active_states(self,
                          tracker: 'DialogueStateTracker'
                          ) -> Dict[Text, float]:
        """Return a bag of active states from the tracker state"""
        state_dict = self.get_parsing_states(tracker)
        state_dict.update(self.get_prev_action_states(tracker))
        state_dict.update(self.get_active_form(tracker))
        return state_dict

    def states_for_tracker_history(self,
                                   tracker: 'DialogueStateTracker'
                                   ) -> List[Dict[Text, float]]:
        """Array of states for each state of the trackers history."""
        return [self.get_active_states(tr) for tr in
                tracker.generate_all_prior_trackers()]

    def slots_for_entities(self, entities):
        if self.store_entities_as_slots:
            slot_events = []
            for s in self.slots:
                if s.auto_fill:
                    matching_entities = [e['value']
                                         for e in entities
                                         if e['entity'] == s.name]
                    if matching_entities:
                        if s.type_name == 'list':
                            slot_events.append(SlotSet(s.name,
                                                       matching_entities))
                        else:
                            slot_events.append(SlotSet(s.name,
                                                       matching_entities[-1]))
            return slot_events
        else:
            return []

    def persist_specification(self, model_path: Text) -> None:
        """Persists the domain specification to storage."""

        domain_spec_path = os.path.join(model_path, 'domain.json')
        utils.create_dir_for_file(domain_spec_path)

        metadata = {
            "states": self.input_states
        }
        utils.dump_obj_as_json_to_file(domain_spec_path, metadata)

    @classmethod
    def load_specification(cls, path: Text) -> Dict[Text, Any]:
        """Load a domains specification from a dumped model directory."""

        metadata_path = os.path.join(path, 'domain.json')
        specification = json.loads(utils.read_file(metadata_path))
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
                "You MUST retrain the policy. " +
                "Detected mismatch in domain specification. " +
                "The following states have been \n"
                "\t - removed: {} \n"
                "\t - added:   {} ".format(missing, additional))
        else:
            return True

    def _slot_definitions(self):
        return {slot.name: slot.persistence_info() for slot in self.slots}

    def as_dict(self):
        # type: () -> Dict[Text, Any]

        additional_config = {
            "store_entities_as_slots": self.store_entities_as_slots}

        return {
            "config": additional_config,
            "intents": [{k: v} for k, v in self.intent_properties.items()],
            "entities": self.entities,
            "slots": self._slot_definitions(),
            "templates": self.templates,
            "actions": self.user_actions,  # class names of the actions
            "forms": self.form_names
        }

    def persist(self, filename: Text) -> None:
        """Write domain to a file."""

        domain_data = self.as_dict()
        utils.dump_obj_as_yaml_to_file(filename, domain_data)

    def persist_clean(self, filename: Text) -> None:
        """Write domain to a file.

         Strips redundant keys with default values."""

        data = self.as_dict()

        for idx, intent_info in enumerate(data["intents"]):
            for name, intent in intent_info.items():
                if intent.get("use_entities"):
                    data["intents"][idx] = name

        for slot in data["slots"].values():
            if slot["initial_value"] is None:
                del slot["initial_value"]
            if slot["auto_fill"]:
                del slot["auto_fill"]
            if slot["type"].startswith('rasa_core.slots'):
                slot["type"] = Slot.resolve_by_type(slot["type"]).type_name

        if data["config"]["store_entities_as_slots"]:
            del data["config"]["store_entities_as_slots"]

        # clean empty keys
        data = {k: v
                for k, v in data.items()
                if v != {} and v != [] and v is not None}

        utils.dump_obj_as_yaml_to_file(filename, data)

    def as_yaml(self):
        domain_data = self.as_dict()
        return utils.dump_obj_as_yaml_to_string(domain_data)

    def intent_config(self, intent_name: Text) -> Dict[Text, Any]:
        """Return the configuration for an intent."""
        return self.intent_properties.get(intent_name, {})

    @utils.lazyproperty
    def intents(self):
        return sorted(self.intent_properties.keys())


class TemplateDomain(Domain):
    pass
