from __future__ import annotations
import inspect
import logging
from os import stat
import os.path
from pathlib import Path
from re import sub
from typing import Optional, Text, Dict, List, Union, Iterable, Tuple
from collections.abc import ValuesView
import copy

from rasa.architecture_prototype.interfaces import ComponentPersistorInterface
from rasa.core.channels import CollectingOutputChannel, UserMessage
from rasa.shared.constants import DEFAULT_DATA_PATH, DEFAULT_DOMAIN_PATH
from rasa.shared.core.constants import DEFAULT_ACTION_NAMES
from rasa.shared.core.domain import Domain, SubState
from rasa.shared.core.events import ActionExecuted, UserUttered, Event
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.constants import ACTION_NAME, ACTION_TEXT, ENTITIES, INTENT, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.features import Features
import rasa.shared.utils.common
import rasa.utils.common
import rasa.core.training


logger = logging.getLogger(__name__)

registry = {}


class GraphComponentMetaclass(type):
    """Metaclass with `name` class property."""

    @property
    def name(cls) -> Text:
        """The name property is a function of the class - its __name__."""
        return cls.__name__

    def __new__(cls, clsname, bases, attrs):
        newclass = super().__new__(cls, clsname, bases, attrs)
        # Every class using this metaclass will be registered automatically when
        # it's imported
        if not inspect.isabstract(newclass):
            registry[newclass.name] = newclass
        return newclass


class GraphComponent(metaclass=GraphComponentMetaclass):
    pass


class ProjectReader(GraphComponent):
    def load_importer(self, project: Text) -> TrainingDataImporter:
        return TrainingDataImporter.load_from_dict(
            domain_path=str(Path(project, DEFAULT_DOMAIN_PATH)),
            training_data_paths=[os.path.join(project, DEFAULT_DATA_PATH)],
        )


class TrainingDataReader(ProjectReader):
    def read(self, project: Text) -> TrainingData:
        importer = self.load_importer(project)
        return rasa.utils.common.run_in_loop(importer.get_nlu_data())


class DomainReader(ProjectReader):
    def __init__(
        self,
        persistor: Optional[ComponentPersistorInterface] = None,
        domain: Optional[Domain] = None,
    ) -> None:
        super().__init__()
        self._persistor = persistor
        self._domain = domain

    @classmethod
    def load(
        cls,
        persistor: Optional[ComponentPersistorInterface] = None,
        resource_name: Optional[Text] = None,
    ) -> DomainReader:
        filename = persistor.get_resource(resource_name, "domain.yml")
        domain = Domain.load(filename)
        return DomainReader(persistor=persistor, domain=domain)

    def read(self, project: Text) -> Domain:
        importer = self.load_importer(project)
        domain = rasa.utils.common.run_in_loop(importer.get_domain())
        target_file = self._persistor.file_for("domain.yml")
        domain.persist(target_file)
        return domain

    def provide(self) -> Domain:
        return self._domain


class StoryGraphReader(ProjectReader):
    def read(self, project: Text) -> StoryGraph:
        importer = self.load_importer(project)

        return rasa.utils.common.run_in_loop(importer.get_stories())


class TrackerGenerator(GraphComponent):
    def generate(
        self, domain: Domain, story_graph: StoryGraph
    ) -> List[TrackerWithCachedStates]:
        generated_coroutine = rasa.core.training.load_data(story_graph, domain,)
        return rasa.utils.common.run_in_loop(generated_coroutine)


class StoryToTrainingDataConverter(GraphComponent):
    """Provides training data for core's NLU pipeline as well as lookup table buildup.

    During training as well as during inference, the given data (i.e. story graph or
    list of messages), will be used to derive a de-duplicated list of all possible
    "prev_action" and "user" sub-states (with either `TEXT` and possibly `ENTITY`
    attributes or only `INTENT` attribute) that could be generated from the given data.

    NOTE: If we wanted to create the training data for the components of the e2e
    featurization pipeline differently from the data that is needed for the lookup
    table (currently, the latter is used for training as well), we simply need to
    move the functionality of this component to e.g. a `StoryToLookUpDataConverter`,
    replace the functionality here with the required training data build up, and
    adapt the graph connections so that the `train` and `predict` steps of the e2e
    run consecutively and use the respective data.
    """

    def convert_for_training(
        self, story_graph: StoryGraph, domain: Domain
    ) -> TrainingData:
        """Creates a list of unique (partial) substates from the domain and story graph.

        Note that partial user substate means all substates with intent as only
        attribute or text plus all possible attributes (except intent).
        """
        lookup_table = E2ELookupTable(handle_collisions=True)

        # collect all action and user (intent-only) substates known from domain
        self.add_sub_states_from_domain(domain=domain, lookup_table=lookup_table)

        # collect all substates we see in the given data
        # TODO: we can skip intent and action with action_name here
        all_events = (
            event for step in story_graph.story_steps for event in step.events
        )
        self.add_sub_states_from_events(events=all_events, lookup_table=lookup_table)

        # make sure that there is at least one user substate with a TEXT to ensure
        # `CountVectorizer` is trained...
        lookup_table.add(Message({TEXT: "hi"}))

        return TrainingData(training_examples=list(lookup_table.values()))

    @staticmethod
    def add_sub_states_from_domain(domain: Domain, lookup_table: E2ELookupTable):
        """Create action and user (intent-only) substates from the domain.

        Args:
          domain: the domain from which we extract the substates
          lookup_table: lookup table to which the substates will be added (as messages)
        """
        # FIXME: which version of action_names ?
        # intial prototype:
        action_texts = domain.action_texts
        action_names = domain.user_actions + DEFAULT_ACTION_NAMES
        # if we use all names (incl forms?) ...
        # TODO: there is no nicer way to just get the action_names
        # action_names = domain.action_names_or_texts[: -len(domain.action_texts)]
        # if set(action_texts).intersection(action_names):
        #     raise NotImplementedError(
        #         "We assumed that domain's action_names_or_texts contains all "
        #         "action names followed by the texts. Apparently that changed. "
        #     )
        for tag, actions in [(ACTION_NAME, action_names), (ACTION_TEXT, action_texts)]:
            for action in actions:
                lookup_table.add(Message({tag: action}))

        for intent in domain.intent_properties.keys():
            lookup_table.add(Message({INTENT: intent}))

    @staticmethod
    def add_sub_states_from_events(
        events: Iterable[Event], lookup_table: E2ELookupTable
    ) -> List[SubState]:
        """Creates all possible action and (partial) user substates from the events.

        Note that partial user substate means all substates with intent as only
        attribute or text plus all possible attributes (except intent).

        Args:
          events: list of events to extract the substate from
          lookup_table: lookup table to which the substates will be added (as messages)
        """
        for event in events:
            if isinstance(event, UserUttered):
                # avoid side effects by making a copy...
                event_copy = copy.deepcopy(event)
                # ... before changing this flag so we get a complete sub-state
                event_copy.use_text_for_featurization = None
                artificial_sub_state = event_copy.as_sub_state()
                # split it up...
                sub_states_from_event = []
                intent = artificial_sub_state.pop(INTENT, None)
                if intent:
                    sub_states_from_event.append({INTENT: intent})
                if len(artificial_sub_state):
                    sub_states_from_event.append(artificial_sub_state)
                    # FIXME: seems we can just remove entities here....
            elif isinstance(event, ActionExecuted):
                sub_states_from_event = []
                sub_state = event.as_sub_state()
                for key in [ACTION_NAME, ACTION_TEXT]:
                    if key in sub_state:
                        sub_states_from_event.append({key: sub_state[key]})
            else:
                sub_states_from_event = []

            for sub_state in sub_states_from_event:
                message = Message(data=sub_state)
                lookup_table.add(message)

    def convert_for_inference(self, tracker: DialogueStateTracker) -> List[Message]:
        """Creates a list of unique (partial) substates from the events in the tracker.

        Note that partial user substate means all substates with intent as only
        attribute or text plus all possible attributes (except intent).

        Args:
          tracker: contains the events from which we want to extract the substates
        Returns:
          a list of messages that wrap the unique substates
        """
        # NOTE: `tracker.applied_events()` doesn't convert any events to a different
        # type and hence just iterating over the events is quicker than "applying"
        # events first and then iterating over results (again).
        lookup_table = E2ELookupTable(handle_collisions=True)
        self.add_sub_states_from_events(tracker.events, lookup_table)
        return list(lookup_table.values())


class E2ELookupTable:
    """A key-value store that stores specific `Messages` only.

    The reason this lookup table behaves as it does is that
    a) our policies only need to pass pass very specific information through the
       tokenization and featurization pipeline
    b) our tokenization and featurization pipeline featurize certain attributes
       independently of each other.

    Therefore, this key-value store can only store messages where exactly one
    attribute is one of `ACTION_NAME`, `ACTION_TEXT`, `TEXT`, or `INTENT`.
    Moreover, it checks that if an `ENTITIES` attribute appears, it is contained in a
    message with `TEXT` (in order to catch bugs due to wrong usage early on).

    Keys are constructed by concatenating the key attribute, which is defined by
    the list of attributes mentioned above, and the actual payload for that attribute
    in the Message. This is to avoid collisions that we otherwise cannot resolve
    afterwards (Example: An intent name also appears as text and interpreted as text,
    it will be "dense featurizable" while the intent name is not)

    The store will resolve collisions on it's own to some degree. Messages with the
    same attributes and the same number of features will be treated as copies (and
    simply not be added again).

    Moverover, if advanced collision handling is enabled, ....
    # FIXME: check if this makes any sense if entities are not used anywhere...
    # FIXME: remove ENTITIES?

    However, it is not capable of resolving all kind of conflicts. If messages contain
    different attributes and different number of features, the lookup table can only
    choose to ignore that or raise an error.

    Args:
      handle_collisions: if set to True, collisions where one Message contains a larger
        or equal number of attributes and of features than the other Message, then
        the collision will be resolved automatically
    """

    KEY_ATTRIBUTES = [ACTION_NAME, ACTION_TEXT, TEXT, INTENT]

    def __init__(self, handle_collisions: bool = True):
        self._table: Dict[Text, SubState] = {}
        self._handle_collisions = handle_collisions
        self._num_collisions_ignored = 0
        self._num_collisions_resolved = 0

    def __repr__(self) -> Text:
        return f"{self.__class__.__name__}({self._table})"

    def __len__(self) -> int:
        return len(self._table)

    def values(self) -> ValuesView:
        return self._table.values()

    @property
    def num_collisions_ignored(self):
        return self._num_collisions_ignored

    @property
    def num_collisions_resolved(self):
        return self._num_collisions_resolved

    @classmethod
    def _build_key(cls, sub_state: SubState) -> Tuple[Text, Text]:
        """Builds the lookup table key for the given substate.

        Expects the message to have exactly one of the `E2ELookupTable.KEY_ATTRIBUTES`.
        Moreover, if `ENTITIES` is an attribute, then `TEXT` must be present.

        Args:
          sub_state: substate we want to store in the lookup table
        Raises:
          ValueErrors if there is more than one key attribute in the given substate
          or if the substate contains `ENTITIES` but no `TEXT`
        """
        attributes = sub_state.keys()
        key_attributes = set(attributes).intersection(cls.KEY_ATTRIBUTES)
        if not key_attributes or len(key_attributes) > 1:
            raise ValueError(
                f"Expected exactly one attribute out of "
                f"{cls.KEY_ATTRIBUTES} but received {attributes}"
            )
        if ENTITIES in attributes and not TEXT in attributes:
            raise ValueError(
                f"Expected entities information only in conjunction with `TEXT` "
                f"but received a substate with {sub_state.keys()}."
            )
        key_attribute = list(key_attributes)[0]
        return (key_attribute, str(sub_state[key_attribute]))

    def add(self, message: Message) -> None:
        """Adds the given message to the lookup table.

        Args:
          message: the message we want to add to the lookup table
        Raises:
          ValueError if we cannot create a key for the given message or if this creates
          a collision that we cannot resolve
        """
        key = self._build_key(sub_state=message.data)
        existing_message = self._table.get(key)
        if existing_message is not None:
            if (
                len(existing_message.features) != len(message.features)
                or existing_message.data.keys() != message.data.keys()
            ):
                if self._handle_collisions:
                    if len(existing_message.features) <= len(message.features) and set(
                        message.data.keys()
                    ) >= set(existing_message.data.keys()):
                        self._table[key] = message
                        self._num_collisions_resolved += 1
                    else:
                        self._num_collisions_ignored += 1

                else:
                    raise ValueError(
                        f"Expected added message to be consistent. {key} already maps to "
                        f"{existing_message}, but we want to add {message} now."
                    )
            else:
                self._num_collisions_ignored += 1
        else:
            self._table[key] = message

    def add_all(self, messages: Iterable[Message]) -> None:
        """Adds the given message to the lookup table.

        Args:
          messages: the messages that we want to add to the lookup table
        Raises:
          ValueError if we cannot create a key for one of the given messages or if
          adding a message creates a collision that we cannot resolve
        """
        for message in messages:
            self.add(message)

    def lookup_features(
        self, sub_state: SubState, attributes: Optional[List[Text]] = None
    ) -> Dict[Text, List[Features]]:
        """Looks up all features for the given substate.

        Note that here we do **not** impose any restrictions on the attributes that
        may be included in the given sub_state.
        There might be be multiple messages in the lookup table that contain features
        relevant for the given substate, e.g. this is the case if `TEXT` and
        `INTENT` are present in the given message.

        Args:
          sub_state: substate for which we want to lookup all relevant Messages
          attributes: if not None, this specifies the list of the attributes of the
            `Features` that we're interested in (i.e. all other `Features` contained
            in the relevant messages will be ignored)
        Returns:
          a dictionary that maps all the (requested) attributes for which we found
          `Features` to a list of these `Features`
        Raises:
          - a `ValueError` if any of the lookup table keys that can be generated from
            the given substate is not contained in the lookup table
          - a `RuntimeError` if an inconsistency in the lookup table is detected;
            more precisely, if features for the same attribute are found in two
            looked up messages
        """
        features = {}
        # get all keys whose values in the lookup table contain features for the
        # given substate
        key_attributes = set(sub_state.keys()).intersection(self.KEY_ATTRIBUTES)
        for key_attribute in key_attributes:
            key = self._build_key(sub_state={key_attribute: sub_state[key_attribute]})
            message = self._table.get(key)
            if not message:
                raise ValueError(
                    f"Unknown key {key}. Cannot retrieve features for substate "
                    f"{sub_state}"
                )
            features_from_message = self.extract_features(
                message, attributes=attributes
            )
            for feat_attribute, feat_value in features_from_message.items():
                existing_values = features.get(feat_attribute)
                if existing_values:
                    raise RuntimeError(
                        f"Feature for attribute {feat_attribute} has already been "
                        f"extracted from a different message stored under a key "
                        f"in {key_attributes} "
                        f"that is different from {key_attribute}. This means there's a "
                        f"redundancy in the lookup table."
                    )
                features[feat_attribute] = feat_value
        return features

    def lookup_message(self, user_text: Text) -> Message:
        """Returns the lookup table entry identified by the given user text.

        Args:
          user_text: the text of a user utterance
        Raises:
          `ValueError` if there is no message associated with the given user text
        """
        key = self.build_key(attribute=TEXT, value=user_text)
        message = self._table.get(key)
        if message is None:
            raise ValueError(f"Expected a message with key {key} in lookup table.")
        return message

    @staticmethod
    def extract_features(
        message: Message, attributes: Optional[List[Text]] = None
    ) -> Dict[Text, List[Features]]:
        """Creates an attribute to features mapping from the given message.

        Args:
          message: the message that possibly contains features
          attributes: if specified, only features whose attributes is in this list
            will be returned
        Returns:
          a dictionary mapping all (relevant) attributes for which we found features
          to a list of these features
        """
        extracted = dict()
        for feat in message.features:
            if attributes is None or feat.attribute in attributes:
                extracted.setdefault(feat.attribute, []).append(feat)
        return extracted


class MessageToE2EFeatureConverter(GraphComponent):
    """Collects featurised messages for use by an e2e policy."""

    def convert(self, messages: Union[TrainingData, List[Message]]) -> E2ELookupTable:
        """Converts messages created by `StoryToTrainingDataConverter` to E2EFeatures.
        """
        # Note that the input messages had been contained in a lookup table in
        # `StoryToTrainingDataConverter. Hence, we don't need to worry about
        # collisions here anymore.
        if isinstance(messages, TrainingData):
            messages = messages.training_examples
        lookup_table = E2ELookupTable()
        for message in messages:
            lookup_table.add(message)
        return lookup_table


class MessageCreator(GraphComponent):
    def __init__(self, message: Optional[UserMessage]) -> None:
        self._message = message

    def create(self) -> Optional[UserMessage]:
        return self._message


class ProjectProvider(GraphComponent):
    def __init__(self, project: Optional[Text]) -> None:
        self._project = project

    def get(self) -> Optional[Text]:
        return self._project


class TrackerLoader(GraphComponent):
    def __init__(self, tracker: DialogueStateTracker) -> None:
        self._tracker = tracker

    def load(self) -> DialogueStateTracker:
        return self._tracker


class NLUMessageConverter(GraphComponent):
    def convert(self, message: Optional[UserMessage]) -> List[Message]:
        if message:
            return [Message.build(message.text)]
        return []


class NLUPredictionToHistoryAdder(GraphComponent):
    def merge(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        initial_user_message: Optional[UserMessage] = None,
        parsed_messages: List[Message] = None,
    ) -> DialogueStateTracker:
        for parsed_message in parsed_messages:
            parse_data = parsed_message.as_dict(only_output_properties=True)

            tracker.update(
                UserUttered(
                    initial_user_message.text,
                    parse_data["intent"],
                    parse_data["entities"],
                    parse_data,
                    input_channel=initial_user_message.input_channel,
                    message_id=initial_user_message.message_id,
                    metadata=initial_user_message.metadata,
                ),
                domain,
            )
        return tracker


class GraphComponentNotFound(Exception):
    pass


def load_graph_component(name: Text):
    if name not in registry:
        raise GraphComponentNotFound()

    return registry[name]
