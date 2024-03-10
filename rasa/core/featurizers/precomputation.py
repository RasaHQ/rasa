from __future__ import annotations
from typing import Optional, Text, Dict, List, Union, Iterable, Any
from collections.abc import ValuesView, KeysView

from rasa.engine.graph import GraphComponent
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.graph import ExecutionContext
from rasa.shared.core.domain import Domain, SubState
from rasa.shared.core.events import ActionExecuted, UserUttered, Event
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.nlu.constants import ACTION_NAME, ACTION_TEXT, INTENT, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.features import Features
import rasa.shared.utils.io

# TODO: make precomputations (MessageContainerForCoreFeaturization) cacheable


class MessageContainerForCoreFeaturization:
    """A key-value store for specific `Messages`.

    This container can be only be used to store messages that contain exactly
    one of the following attributes: `ACTION_NAME`, `ACTION_TEXT`, `TEXT`, or `INTENT`.
    A combination of the key attribute and the corresponding value will be used as
    key for the respective message.

    Background/Motivation:
    - Our policies only require these attributes to be tokenized and/or featurized
      via NLU graph components, which is why we don't care about storing anything else.
    - Our tokenizers and featurizers work independently for each attribute,
      which is why we can separate them and ask for "exactly one" of the key
      attributes.
    - Our tokenizers add attributes (e.g. token sequences) and not just `Features`,
      which is why we need messages and why we allow messages to contain more than
      just the key attributes.
    - Due to the way we use this datastructure, it won't contain all features that the
      policies need (cf. `rasa.core.featurizers.SingleStateFeaturizer`) and sometimes
      the messages will contain no features at all, which is the motivation for the
      name of this class.
    - Values for different attributes might coincide (e.g. 'greet' can appear as user
      text as well as name of an intent), but attributes are not all tokenized and
      featurized in the same way, which is why we use the combination of key attribute
      and value to identify a message.

    Usage:
    - At the start of core's featurization pipeline, we use this container to
      de-duplicate the given story data during training (e.g. "Hello" might appear very
      often but it will end up in the training data only once) and to de-duplicate
      the data given in the tracker (e.g. if a text appears repeatedly in the
      dialogue, it will only be featurized once later).
      See: `rasa.core.featurizers.precomputation.CoreFeaturizationInputConverter`.
    - At the end of core's featurization pipeline, we wrap all resulting
      (training data) messages into this container again.
      See: `rasa.core.featurizers.precomputation.CoreFeaturizationCollector`.
    """

    KEY_ATTRIBUTES = [ACTION_NAME, ACTION_TEXT, TEXT, INTENT]  # noqa: RUF012

    def __init__(self) -> None:
        """Creates an empty container for precomputations."""
        self._table: Dict[Text, Dict[Text, Message]] = {
            key: {} for key in self.KEY_ATTRIBUTES
        }
        self._num_collisions_ignored = 0

    def fingerprint(self) -> Text:
        """Fingerprint the container.

        Returns:
            hex string as a fingerprint of the container.
        """
        message_fingerprints = [
            message.fingerprint() for message in self.all_messages()
        ]
        return rasa.shared.utils.io.deep_container_fingerprint(message_fingerprints)

    def __repr__(self) -> Text:
        return f"{self.__class__.__name__}({self._table})"

    def __len__(self) -> int:
        return sum(
            len(key_attribute_table) for key_attribute_table in self._table.values()
        )

    def messages(self, key_attribute: Optional[Text] = None) -> ValuesView:
        """Returns a view of all messages."""
        if key_attribute not in self._table:
            raise ValueError(
                f"Expected key attribute (i.e. one of {self.KEY_ATTRIBUTES}) "
                f"but received {key_attribute}."
            )
        return self._table[key_attribute].values()

    def all_messages(self) -> List[Message]:
        """Returns a list containing all messages."""
        return [
            message
            for key_attribute_table in self._table.values()
            for message in key_attribute_table.values()
        ]

    def keys(self, key_attribute: Text) -> KeysView:
        """Returns a view of the value keys for the given key attribute."""
        if key_attribute not in self._table:
            raise ValueError(
                f"Expected key attribute (i.e. one of {self.KEY_ATTRIBUTES}) "
                f"but received {key_attribute}."
            )
        return self._table[key_attribute].keys()

    @property
    def num_collisions_ignored(self) -> int:
        """Returns the number of collisions that have been ignored."""
        return self._num_collisions_ignored

    def add(self, message_with_one_key_attribute: Message) -> None:
        """Adds the given message if it is not already present.

        Args:
          message_with_one_key_attribute: The message we want to add to the lookup
            table. It must have exactly one key attribute.

        Raises:
          `ValueError` if the given message does not contain exactly one key
          attribute or if there is a collision with a message that has a different
          hash value
        """
        # extract the key pair
        attributes = message_with_one_key_attribute.data.keys()
        key_attributes = set(attributes).intersection(self.KEY_ATTRIBUTES)
        if not key_attributes or len(key_attributes) != 1:
            raise ValueError(
                f"Expected exactly one attribute out of "
                f"{self.KEY_ATTRIBUTES} but received {len(attributes)} attributes "
                f"({attributes})."
            )
        key_attribute = list(key_attributes)[0]  # noqa: RUF015
        key_value = str(message_with_one_key_attribute.data[key_attribute])
        # extract the message
        existing_message = self._table[key_attribute].get(key_value)
        if existing_message is not None:
            if hash(existing_message) != hash(message_with_one_key_attribute):
                raise ValueError(
                    f"Expected added message to be consistent. "
                    f"({key_attribute}, {key_value}) already maps "
                    f"to {existing_message}, but we want to add "
                    f"{message_with_one_key_attribute} now."
                )
            else:
                self._num_collisions_ignored += 1
        else:
            self._table[key_attribute][key_value] = message_with_one_key_attribute

    def add_all(self, messages_with_one_key_attribute: List[Message]) -> None:
        """Adds the given messages.

        Args:
          messages_with_one_key_attribute: The messages that we want to add.
            Each one must have exactly one key attribute.

        Raises:
          `ValueError` if we cannot create a key for the given message or if there is
          a collisions with a message that has a different hash value
        """
        for message in messages_with_one_key_attribute:
            self.add(message)

    def collect_features(
        self, sub_state: SubState, attributes: Optional[Iterable[Text]] = None
    ) -> Dict[Text, List[Features]]:
        """Collects features for all attributes in the given substate.

        There might be be multiple messages in the container that contain features
        relevant for the given substate, e.g. this is the case if `TEXT` and
        `INTENT` are present in the given substate. All of those messages will be
        collected and their features combined.

        Args:
          sub_state: substate for which we want to extract the relevent features
          attributes: if not `None`, this specifies the list of the attributes of the
            `Features` that we're interested in (i.e. all other `Features` contained
            in the relevant messages will be ignored)

        Returns:
          a dictionary that maps all the (requested) attributes to a list of `Features`

        Raises:
          `ValueError`: if there exists some key pair (i.e. key attribute and
            corresponding value) from the given substate cannot be found
          `RuntimeError`: if features for the same attribute are found in two
            different messages that are associated with the given substate
        """
        # If we specify a list of attributes, then we want a dict with one entry
        # for each attribute back - even if the corresponding list of features is empty.
        features: Dict[Text, List[Features]] = (
            dict()
            if attributes is None
            else {attribute: [] for attribute in attributes}
        )
        # collect all relevant key attributes
        key_attributes = set(sub_state.keys()).intersection(self.KEY_ATTRIBUTES)
        for key_attribute in key_attributes:
            key_value = str(sub_state[key_attribute])
            message = self._table[key_attribute].get(key_value)
            if not message:
                raise ValueError(
                    f"Unknown key ({key_attribute},{key_value}). Cannot retrieve "
                    f"features for substate {sub_state}"
                )
            features_from_message = Features.groupby_attribute(
                message.features, attributes=attributes
            )
            for feat_attribute, feat_value in features_from_message.items():
                existing_values = features.get(feat_attribute)
                # Note: the following if-s are needed because if we specify a list of
                # attributes then `features_from_message` will contain one entry per
                # attribute even if the corresponding feature list is empty.
                if feat_value and existing_values:
                    raise RuntimeError(
                        f"Feature for attribute {feat_attribute} has already been "
                        f"extracted from a different message stored under a key "
                        f"in {key_attributes} "
                        f"that is different from {key_attribute}. This means there's a "
                        f"redundancy in the message container."
                    )
                if feat_value:
                    features[feat_attribute] = feat_value
        return features

    def lookup_message(self, user_text: Text) -> Message:
        """Returns a message that contains the given user text.

        Args:
          user_text: the text of a user utterance
        Raises:
          `ValueError` if there is no message associated with the given user text
        """
        message = self._table[TEXT].get(user_text)
        if message is None:
            raise ValueError(
                f"Expected a message with key ({TEXT}, {user_text}) in lookup table."
            )
        return message

    def derive_messages_from_domain_and_add(self, domain: Domain) -> None:
        """Adds all lookup table entries that can be derived from the domain.

        That is, all action names, action texts, and intents defined in the domain
        will be turned into a (separate) messages and added to this lookup table.

        Args:
          domain: the domain from which we extract the substates
        """
        if (
            domain.action_texts
            and domain.action_names_or_texts[-len(domain.action_texts) :]
            != domain.action_texts
        ):
            raise NotImplementedError(
                "We assumed that domain's `action_names_or_texts` start with a list of "
                "all action names, followed by the action texts. "
                "Please update the code to grab the action_name and action_texts from "
                "the domain correctly."
            )
        action_texts = domain.action_texts
        action_names = domain.action_names_or_texts[
            slice(0, -len(domain.action_texts) if domain.action_texts else None)
        ]

        for key_attribute, actions in [
            (ACTION_NAME, action_names),
            (ACTION_TEXT, action_texts),
        ]:
            for action in actions:
                self.add(Message({key_attribute: action}))

        for intent in domain.intent_properties.keys():
            self.add(Message({INTENT: intent}))

    def derive_messages_from_events_and_add(self, events: Iterable[Event]) -> None:
        """Adds all relevant messages that can be derived from the given events.

        That is, each action name, action text, user text and intent that can be
        found in the given events will be turned into a (separate) message and added
        to this container.

        Args:
          events: list of events to extract the substate from
        """
        for event in events:
            key_value_list = []
            if isinstance(event, UserUttered):
                key_value_list = [(TEXT, event.text), (INTENT, event.intent_name)]
            elif isinstance(event, ActionExecuted):
                key_value_list = [
                    (ACTION_TEXT, event.action_text),
                    (ACTION_NAME, event.action_name),
                ]
            for key, value in key_value_list:
                if value is not None:
                    self.add(Message(data={key: value}))


class CoreFeaturizationInputConverter(GraphComponent):
    """Provides data for the featurization pipeline.

    During training as well as during inference, the converter de-duplicates the given
    data (i.e. story graph or list of messages) such that each text and intent from a
    user message and each action name and action text appears exactly once.
    """

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> CoreFeaturizationInputConverter:
        """Creates a new instance (see parent class for full docstring)."""
        return cls()

    def convert_for_training(
        self, domain: Domain, story_graph: StoryGraph
    ) -> TrainingData:
        """Creates de-duplicated training data.

        Each possible user text and intent and each action name and action text
        that can be found in the given domain and story graph appears exactly once
        in the resulting training data. Moreover, each item is contained in a separate
        messsage.

        Args:
           domain: the domain
           story_graph: a story graph
        Returns:
           training data
        """
        container = MessageContainerForCoreFeaturization()

        # collect all action and user (intent-only) substates known from domain
        container.derive_messages_from_domain_and_add(domain=domain)

        # collect all substates we see in the given data
        all_events = (
            event
            for step in story_graph.story_steps
            for event in step.events
            if isinstance(event, UserUttered)
            # because all action names and texts are known to the domain
        )
        container.derive_messages_from_events_and_add(events=all_events)

        # Reminder: in case of complex recipes that train CountVectorizers, we'll have
        # to make sure that there is at least one user substate with a TEXT to ensure
        # `CountVectorizer` is trained...

        return TrainingData(training_examples=container.all_messages())

    def convert_for_inference(self, tracker: DialogueStateTracker) -> List[Message]:
        """Creates a list of messages containing single user and action attributes.

        Each possible user text and intent and each action name and action text
        that can be found in the events of the given tracker will appear exactly once
        in the resulting messages. Moreover, each item is contained in a separate
        messsage.

        Args:
          tracker: a dialogue state tracker containing events
        Returns:
          a list of messages
        """
        # Note: `tracker.applied_events()` doesn't convert any events to a different
        # type and hence just iterating over the events is quicker than "applying"
        # events first and then iterating over results (again).
        container = MessageContainerForCoreFeaturization()
        container.derive_messages_from_events_and_add(tracker.events)
        return container.all_messages()


class CoreFeaturizationCollector(GraphComponent):
    """Collects featurized messages for use by a policy."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> CoreFeaturizationCollector:
        """Creates a new instance (see parent class for full docstring)."""
        return cls()

    def collect(
        self, messages: Union[TrainingData, List[Message]]
    ) -> MessageContainerForCoreFeaturization:
        """Collects messages."""
        if isinstance(messages, TrainingData):
            messages = messages.training_examples
        # Note that the input messages had been contained in a lookup table in
        # `StoryToTrainingDataConverter. Hence, we don't need to worry about
        # collisions here anymore.
        container = MessageContainerForCoreFeaturization()
        for message in messages:
            container.add(message)
        return container
