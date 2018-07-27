from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import collections
import io
import json
import logging
import os

import numpy as np
import pkg_resources
from pykwalify.errors import SchemaError
from six import string_types
from six import with_metaclass
from typing import Dict, Tuple, Any
from typing import List
from typing import Optional
from typing import Text

from rasa_core import utils
from rasa_core.actions import Action
from rasa_core.actions.action import (ActionListen, ActionRestart,
                                      ActionDefaultFallback)
from rasa_core.actions.factories import (
    action_factory_by_name,
    ensure_action_name_uniqueness)
from rasa_core.slots import Slot
from rasa_core.trackers import DialogueStateTracker, SlotSet
from rasa_core.utils import read_file, read_yaml_string

logger = logging.getLogger(__name__)

PREV_PREFIX = 'prev_'


def check_domain_sanity(domain):
    """Makes sure the domain is properly configured.

    Checks the settings and checks if there are duplicate actions,
    intents, slots and entities."""

    def get_duplicates(my_items):
        """Returns a list of duplicate items in my_items."""
        return [item
                for item, count in collections.Counter(my_items).items()
                if count > 1]

    def get_exception_message(duplicates):
        """Returns a message given a list of error locations.

        Duplicates has the format of (duplicate_actions [List], name [Text]).
        :param duplicates:
        :return: """

        msg = ""
        for d, name in duplicates:
            if d:
                if msg:
                    msg += "\n"
                msg += ("Duplicate {0} in domain. "
                        "These {0} occur more than once in "
                        "the domain: {1}".format(name, ", ".join(d)))
        return msg

    duplicate_actions = get_duplicates([a for a in domain.actions])
    duplicate_intents = get_duplicates([i for i in domain.intents])
    duplicate_slots = get_duplicates([s.name for s in domain.slots])
    duplicate_entities = get_duplicates([e for e in domain.entities])

    if duplicate_actions or \
            duplicate_intents or \
            duplicate_slots or \
            duplicate_entities:
        raise Exception(get_exception_message([
            (duplicate_actions, "actions"),
            (duplicate_intents, "intents"),
            (duplicate_slots, "slots"),
            (duplicate_entities, "entitites")]))


class Domain(with_metaclass(abc.ABCMeta, object)):
    """The domain specifies the universe in which the bot's policy acts.

    A Domain subclass provides the actions the bot can take, the intents
    and entities it can recognise"""

    DEFAULT_ACTIONS = [ActionListen(), ActionRestart(),
                       ActionDefaultFallback()]

    def __init__(self, store_entities_as_slots=True,
                 restart_intent="restart"):
        self.store_entities_as_slots = store_entities_as_slots
        self.restart_intent = restart_intent

    @utils.lazyproperty
    def num_actions(self):
        """Returns the number of available actions."""

        # noinspection PyTypeChecker
        return len(self.actions)

    @utils.lazyproperty
    def action_names(self):
        # type: () -> List[Text]
        """Returns the name of available actions."""

        return [a.name() for a in self.actions]

    @utils.lazyproperty
    def action_map(self):
        # type: () -> Dict[Text, Tuple[int, Action]]
        """Provides a mapping from action names to indices and actions."""
        return {a.name(): (i, a) for i, a in enumerate(self.actions)}

    @utils.lazyproperty
    def num_states(self):
        """Number of used input states for the action prediction."""

        return len(self.input_states)

    def action_for_name(self, action_name):
        # type: (Text) -> Optional[Action]
        """Looks up which action corresponds to this action name."""

        if action_name in self.action_map:
            return self.action_map.get(action_name)[1]
        else:
            self._raise_action_not_found_exception(action_name)

    def action_for_index(self, index):
        """Integer index corresponding to an actions index in the action list.

        This method resolves the index to the actions name."""

        if len(self.actions) <= index or index < 0:
            raise Exception(
                    "Can not access action at index {}. "
                    "Domain has {} actions.".format(index, len(self.actions)))
        return self.actions[index]

    def index_for_action(self, action_name):
        # type: (Text) -> Optional[int]
        """Looks up which action index corresponds to this action name"""

        if action_name in self.action_map:
            return self.action_map.get(action_name)[0]
        else:
            self._raise_action_not_found_exception(action_name)

    def _raise_action_not_found_exception(self, action_name):
        actions = "\n".join(["\t - {}".format(a)
                             for a in sorted(self.action_map)])
        raise Exception(
                "Can not access action '{}', "
                "as that name is not a registered action for this domain. "
                "Available actions are: \n{}".format(action_name, actions))

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

        return [PREV_PREFIX + a.name()
                for a in self.actions]

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

    def index_of_state(self, state_name):
        # type: (Text) -> Optional[int]
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
            self.prev_action_states

    def get_parsing_states(self, tracker):
        # type: (DialogueStateTracker) -> Dict[Text, float]

        state_dict = {}

        # Set all found entities with the state value 1.0, unless they should
        # be ignored for the current intent
        for entity in tracker.latest_message.entities:
            intent_name = tracker.latest_message.intent.get("name")
            should_use_entity = self._intents[intent_name]['use_entities']
            if should_use_entity:
                key = "entity_{0}".format(entity["entity"])
                state_dict[key] = 1.0

        # Set all set slots with the featurization of the stored value
        for key, slot in tracker.slots.items():
            if slot is not None:
                for i, slot_value in enumerate(slot.as_feature()):
                    if slot_value != 0:
                        slot_id = "slot_{}_{}".format(key, i)
                        state_dict[slot_id] = slot_value

        latest_msg = tracker.latest_message

        if "intent_ranking" in latest_msg.parse_data:
            for intent in latest_msg.parse_data["intent_ranking"]:
                if intent.get("name"):
                    intent_id = "intent_{}".format(intent["name"])
                    state_dict[intent_id] = intent["confidence"]

        elif latest_msg.intent.get("name"):
            intent_id = "intent_{}".format(latest_msg.intent["name"])
            state_dict[intent_id] = latest_msg.intent.get("confidence", 1.0)

        return state_dict

    def get_prev_action_states(self, tracker):
        # type: (DialogueStateTracker) -> Dict[Text, float]
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

    def get_active_states(self, tracker):
        # type: (DialogueStateTracker) -> Dict[Text, float]
        """Return a bag of active states from the tracker state"""
        state_dict = self.get_parsing_states(tracker)
        state_dict.update(self.get_prev_action_states(tracker))
        return state_dict

    def states_for_tracker_history(self, tracker):
        # type: (DialogueStateTracker) -> List[Dict[Text, float]]
        """Array of states for each state of the trackers history."""
        return [self.get_active_states(tr) for tr in
                tracker.generate_all_prior_trackers()]

    def slots_for_entities(self, entities):
        if self.store_entities_as_slots:
            slot_events = []
            for s in self.slots:
                matching_entities = [e['value']
                                     for e in entities
                                     if e['entity'] == s.name]
                if matching_entities:
                    if s.type_name == 'list':
                        slot_events.append(SlotSet(s.name, matching_entities))
                    else:
                        slot_events.append(SlotSet(s.name,
                                                   matching_entities[-1]))
            return slot_events
        else:
            return []

    def persist(self, filename):
        raise NotImplementedError

    @classmethod
    def load(cls, filename):
        raise NotImplementedError

    def persist_specification(self, model_path):
        # type: (Text) -> None
        """Persists the domain specification to storage."""

        domain_spec_path = os.path.join(model_path, 'domain.json')
        utils.create_dir_for_file(domain_spec_path)

        metadata = {
            "states": self.input_states
        }
        utils.dump_obj_as_json_to_file(domain_spec_path, metadata)

    @classmethod
    def load_specification(cls, path):
        # type: (Text) -> Dict[Text, Any]
        """Load a domains specification from a dumped model directory."""

        matadata_path = os.path.join(path, 'domain.json')
        with io.open(matadata_path) as f:
            specification = json.loads(f.read())
        return specification

    def compare_with_specification(self, path):
        # type: (Text) -> bool
        """Compares the domain spec of the current and the loaded domain.

        Throws exception if the loaded domain specification is different
        to the current domain are different."""

        loaded_domain_spec = self.load_specification(path)
        states = loaded_domain_spec["states"]
        if states != self.input_states:
            missing = ",".join(set(states) - set(self.input_states))
            additional = ",".join(set(self.input_states) - set(states))
            raise Exception(
                    "Domain specification has changed. "
                    "You MUST retrain the policy. " +
                    "Detected mismatch in domain specification. " +
                    "The following states have been \n"
                    "\t - removed: {} \n"
                    "\t - added:   {} ".format(missing, additional))
        else:
            return True

    # Abstract Methods : These have to be implemented in any domain subclass
    @abc.abstractproperty
    def slots(self):
        # type: () -> List[Slot]
        """Domain subclass must provide a list of slots"""
        pass

    @abc.abstractproperty
    def entities(self):
        # type: () -> List[Text]
        raise NotImplementedError(
                "domain must provide a list of entities")

    @abc.abstractproperty
    def intents(self):
        # type: () -> List[Text]
        raise NotImplementedError(
                "domain must provide a list of intents")

    @abc.abstractproperty
    def actions(self):
        # type: () -> List[Action]
        raise NotImplementedError(
                "domain must provide a list of possible actions")

    @abc.abstractproperty
    def templates(self):
        # type: () -> List[Dict[Text, Any]]
        raise NotImplementedError(
                "domain must provide a dictionary of response templates")


class TemplateDomain(Domain):

    @classmethod
    def load(cls, filename, action_factory=None):
        if not os.path.isfile(filename):
            raise Exception(
                    "Failed to load domain specification from '{}'. "
                    "File not found!".format(os.path.abspath(filename)))
        return cls.load_from_yaml(read_file(filename), action_factory=action_factory)

    @classmethod
    def load_from_yaml(cls, yaml, action_factory=None):
        cls.validate_domain_yaml(yaml)
        data = read_yaml_string(yaml)
        utter_templates = cls.collect_templates(data.get("templates", {}))
        if not action_factory:
            action_factory = data.get("action_factory", None)
        slots = cls.collect_slots(data.get("slots", {}))
        additional_arguments = data.get("config", {})
        intents = cls.collect_intents(data.get("intents", {}))
        return cls(
            intents,
            data.get("entities", []),
            slots,
            utter_templates,
            data.get("actions", []),
            data.get("action_names", []),
            action_factory,
            **additional_arguments
        )

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
            raise ValueError("Failed to validate your domain yaml '{}'. "
                             "Make sure the file is correct, to do so"
                             "take a look at the errors logged during "
                             "validation previous to this exception. "
                             "".format(os.path.abspath(input)))

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
    def collect_intents(intent_list):
        intents = {}
        for intent in intent_list:
            if isinstance(intent, dict):
                intents.update(intent)
            else:
                intents.update({intent: {'use_entities': True}})
        return intents

    @staticmethod
    def collect_templates(yml_templates):
        # type: (Dict[Text, List[Any]]) -> Dict[Text, List[Dict[Text, Any]]]
        """Go through the templates and make sure they are all in dict format"""

        templates = {}
        for template_key, template_variations in yml_templates.items():
            validated_variations = []
            for t in template_variations:
                # templates can either directly be strings or a dict with
                # options we will always create a dict out of them
                if isinstance(t, string_types):
                    validated_variations.append({"text": t})
                elif "text" not in t:
                    raise Exception("Utter template '{}' needs to contain"
                                    "'- text: ' attribute to be a proper"
                                    "template".format(template_key))
                else:
                    validated_variations.append(t)
            templates[template_key] = validated_variations
        return templates

    def __init__(self, intents, entities, slots, templates, action_classes,
                 action_names, action_factory, **kwargs):
        self._intents = intents
        self._entities = entities
        self._slots = slots
        self._templates = templates
        self._action_classes = action_classes
        self._action_names = action_names
        self._factory_name = action_factory
        self._actions = self.instantiate_actions(
                action_factory, action_classes, action_names, templates)
        super(TemplateDomain, self).__init__(**kwargs)

    @staticmethod
    def instantiate_actions(factory_name, action_classes, action_names,
                            templates):
        action_factory = action_factory_by_name(factory_name)
        custom_actions = action_factory(action_classes, action_names, templates)
        actions = Domain.DEFAULT_ACTIONS[:] + custom_actions
        ensure_action_name_uniqueness(actions)
        return actions

    def _slot_definitions(self):
        return {slot.name: slot.persistence_info() for slot in self.slots}

    def as_dict(self):
        additional_config = {
            "store_entities_as_slots": self.store_entities_as_slots}
        action_names = self.action_names[len(Domain.DEFAULT_ACTIONS):]
        domain_data = {
            "config": additional_config,
            "intents": [{k: v} for k, v in self._intents.items()],
            "entities": self.entities,
            "slots": self._slot_definitions(),
            "templates": self.templates,
            "actions": self._action_classes,  # class names of the actions
            "action_names": action_names,  # names in stories
            "action_factory": self._factory_name
        }
        return domain_data

    def persist(self, filename):
        domain_data = self.as_dict()
        utils.dump_obj_as_yaml_to_file(filename, domain_data)

    def as_yaml(self):
        domain_data = self.as_dict()
        return utils.dump_obj_as_yaml_to_string(domain_data)

    @utils.lazyproperty
    def templates(self):
        return self._templates

    @utils.lazyproperty
    def slots(self):
        return self._slots

    @utils.lazyproperty
    def intents(self):
        return sorted(self._intents.keys())

    @utils.lazyproperty
    def entities(self):
        return self._entities

    @utils.lazyproperty
    def actions(self):
        return self._actions
