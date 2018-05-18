from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import shutil

import typing
from six import string_types
from typing import Text, List, Optional, Callable, Any, Dict, Union

from rasa_core import training
from rasa_core.channels import UserMessage, InputChannel, OutputChannel
from rasa_core.domain import TemplateDomain, Domain, check_domain_sanity
from rasa_core.events import Event
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.policies import Policy
from rasa_core.policies.ensemble import SimplePolicyEnsemble, PolicyEnsemble
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.processor import MessageProcessor
from rasa_core.tracker_store import InMemoryTrackerStore, TrackerStore
from rasa_core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.interpreter import NaturalLanguageInterpreter as NLI


class Agent(object):
    """Public interface for common things to do.

     This includes e.g. train an assistant, or handle messages
     with an assistant."""

    def __init__(
            self,
            domain,  # type: Union[Text, Domain]
            policies=None,  # type: Union[PolicyEnsemble, List[Policy], None]
            interpreter=None,  # type: Union[NLI, Text, None]
            tracker_store=None  # type: Optional[TrackerStore]
    ):
        self.domain = self._create_domain(domain)
        self.policy_ensemble = self._create_ensemble(policies)
        self.interpreter = NaturalLanguageInterpreter.create(interpreter)
        self.tracker_store = self.create_tracker_store(
                tracker_store, self.domain)

    @classmethod
    def load(cls,
             path,  # type: Text
             interpreter=None,  # type: Union[NLI, Text, None]
             tracker_store=None,  # type: Optional[TrackerStore]
             action_factory=None  # type: Optional[Text]
             ):
        # type: (Text, Any, Optional[TrackerStore]) -> Agent
        """Load a persisted model from the passed path."""

        if path is None:
            raise ValueError("No domain path specified.")

        if os.path.isfile(path):
            raise ValueError("You are trying to load a MODEL from a file "
                             "('{}'), which is not possible. \n"
                             "The persisted path should be a directory "
                             "containing the various model files. \n\n"
                             "If you want to load training data instead of "
                             "a model, use `agent.load_data(...)` "
                             "instead.".format(path))

        ensemble = PolicyEnsemble.load(path)
        domain = TemplateDomain.load(os.path.join(path, "domain.yml"),
                                     action_factory)
        # ensures the domain hasn't changed between test and train
        domain.compare_with_specification(path)
        _interpreter = NaturalLanguageInterpreter.create(interpreter)
        _tracker_store = cls.create_tracker_store(tracker_store, domain)

        return cls(domain, ensemble, _interpreter, _tracker_store)

    def handle_message(
            self,
            text_message,  # type: Text
            message_preprocessor=None,  # type: Optional[Callable[[Text], Text]]
            output_channel=None,  # type: Optional[OutputChannel]
            sender_id=UserMessage.DEFAULT_SENDER_ID  # type: Optional[Text]
    ):
        # type: (...) -> Optional[List[Text]]
        """Handle a single message.

        If a message preprocessor is passed, the message will be passed to that
        function first and the return value is then used as the
        input for the dialogue engine.

        The return value of this function depends on the `output_channel`. If
        the output channel is not set, set to `None`, or set
        to `CollectingOutputChannel` this function will return the messages
        the bot wants to respond.

        :Example:

            >>> from rasa_core.agent import Agent
            >>> agent = Agent.load("examples/restaurantbot/models/dialogue",
            ... interpreter="examples/restaurantbot/models/nlu/current")
            >>> agent.handle_message("hello")
            [u'how can I help you?']

        """

        processor = self._create_processor(message_preprocessor)
        return processor.handle_message(
                UserMessage(text_message, output_channel, sender_id))

    def start_message_handling(
            self,
            text_message,   # type: Text
            sender_id=UserMessage.DEFAULT_SENDER_ID  # type: Optional[Text]
    ):
        # type: (...) -> Dict[Text, Any]
        """Start to process a messages, returning the next action to take. """

        processor = self._create_processor()
        return processor.start_message_handling(
                UserMessage(text_message, None, sender_id))

    def continue_message_handling(
            self,
            sender_id,  # type: Text
            executed_action,   # type: Text
            events   # type: List[Event]
    ):
        # type: (...) -> Dict[Text, Any]
        """Continue to process a messages.

        Predicts the next action to take by the caller"""

        processor = self._create_processor()
        return processor.continue_message_handling(sender_id,
                                                   executed_action,
                                                   events)

    def handle_channel(
            self,
            input_channel,  # type: InputChannel
            message_preprocessor=None   # type: Optional[Callable[[Text], Text]]
    ):
        # type: (...) -> None
        """Handle messages coming from the channel."""

        processor = self._create_processor(message_preprocessor)
        processor.handle_channel(input_channel)

    def toggle_memoization(
            self,
            activate   # type: bool
    ):
        # type: (...) -> None
        """Toggles the memoization on and off.

        If a memoization policy is present in the ensemble, this will toggle
        the prediction of that policy. When set to `false` the Memoization
        policies present in the policy ensemble will not make any predictions.
        Hence, the prediction result from the ensemble always needs to come
        from a different policy (e.g. `KerasPolicy`). Useful to test prediction
        capabilities of an ensemble when ignoring memorized turns from the
        training data."""

        for p in self.policy_ensemble.policies:
            # explicitly ignore inheritance (e.g. augmented memoization policy)
            if type(p) == MemoizationPolicy:
                p.toggle(activate)

    def load_data(self,
                  resource_name,  # type: Text
                  remove_duplicates=True,  # type: bool
                  augmentation_factor=20,  # type: int
                  max_number_of_trackers=2000,  # type: int
                  tracker_limit=None,  # type: Optional[int]
                  use_story_concatenation=True  # type: bool
                  ):
        # type: (...) -> List[DialogueStateTracker]
        """Load training data from a resource."""

        return training.load_data(resource_name, self.domain, remove_duplicates,
                                  augmentation_factor, max_number_of_trackers,
                                  tracker_limit, use_story_concatenation)

    def train(self,
              training_trackers,  # type: List[DialogueStateTracker]
              **kwargs  # type: **Any
              ):
        # type: (...) -> None
        """Train the policies / policy ensemble using dialogue data from file.

            :param training_trackers: trackers to train on
            :param kwargs: additional arguments passed to the underlying ML
                           trainer (e.g. keras parameters)
        """

        # deprecation tests
        if kwargs.get('featurizer') or kwargs.get('max_history'):
            raise Exception("Passing `featurizer` and `max_history` "
                            "to `agent.train(...)` is not supported anymore. "
                            "Pass appropriate featurizer "
                            "directly to the policy instead. More info "
                            "https://core.rasa.com/migrations.html#x-to-0-9-0")

        # TODO: DEPRECATED - remove in version 0.10
        if isinstance(training_trackers, string_types):
            # the user most likely passed in a file name to load training
            # data from
            logger.warning("Passing a file name to `agent.train(...)` is "
                           "deprecated. Rather load the data with "
                           "`data = agent.load_data(file_name)` and pass it "
                           "to `agent.train(data)`.")
            training_trackers = self.load_data(training_trackers)

        logger.debug("Agent trainer got kwargs: {}".format(kwargs))
        check_domain_sanity(self.domain)

        self.policy_ensemble.train(training_trackers, self.domain,
                                   **kwargs)

    def train_online(self,
                     training_trackers,  # type: List[DialogueStateTracker]
                     input_channel=None,  # type: Optional[InputChannel]
                     max_visual_history=3,  # type: int
                     **kwargs  # type: **Any
                     ):
        # type: (...) -> None
        from rasa_core.policies.online_trainer import OnlinePolicyEnsemble
        """Train a policy ensemble in online learning mode."""

        if not self.interpreter:
            raise ValueError(
                    "When using online learning, you need to specify "
                    "an interpreter for the agent to use.")

        # TODO: DEPRECATED - remove in version 0.10
        if isinstance(training_trackers, string_types):
            # the user most likely passed in a file name to load training
            # data from
            logger.warning("Passing a file name to `agent.train_online(...)` "
                           "is deprecated. Rather load the data with "
                           "`data = agent.load_data(file_name)` and pass it "
                           "to `agent.train_online(data)`.")
            training_trackers = self.load_data(training_trackers)

        logger.debug("Agent online trainer got kwargs: {}".format(kwargs))
        check_domain_sanity(self.domain)

        self.policy_ensemble.train(training_trackers, self.domain, **kwargs)

        ensemble = OnlinePolicyEnsemble(self.policy_ensemble,
                                        training_trackers,
                                        max_visual_history)

        ensemble.run_online_training(self.domain, self.interpreter,
                                     input_channel)

    @staticmethod
    def _clear_model_directory(model_path):
        # type: (Text) -> None
        """Remove existing files from model directory.

        Only removes files if the directory seems to contain a previously
        persisted model. Otherwise does nothing to avoid deleting
        `/` by accident."""

        if not os.path.exists(model_path):
            return

        domain_spec_path = os.path.join(model_path, 'policy_metadata.json')
        # check if there were a model before
        if os.path.exists(domain_spec_path):
            logger.info("Model directory {} exists and contains old "
                        "model files. All files will be overwritten."
                        "".format(model_path))
            shutil.rmtree(model_path)
        else:
            logger.debug("Model directory {} exists, but does not contain "
                         "all old model files. Some files might be "
                         "overwritten.".format(model_path))

    def persist(self, model_path):
        # type: (Text) -> None
        """Persists this agent into a directory for later loading and usage."""

        self._clear_model_directory(model_path)

        self.policy_ensemble.persist(model_path)
        self.domain.persist(os.path.join(model_path, "domain.yml"))
        self.domain.persist_specification(model_path)

        logger.info("Persisted model to '{}'"
                    "".format(os.path.abspath(model_path)))

    def visualize(self,
                  resource_name,  # type: Text
                  output_file,  # type: Text
                  max_history,  # type: int
                  nlu_training_data=None,  # type: Optional[Text]
                  should_merge_nodes=True,  # type: bool
                  fontsize=12  # type: int
                  ):
        # type: (...) -> None
        from rasa_core.training.visualization import visualize_stories
        from rasa_core.training.dsl import StoryFileReader
        """Visualize the loaded training data from the resource."""

        story_steps = StoryFileReader.read_from_folder(resource_name,
                                                       self.domain)
        visualize_stories(story_steps, self.domain, output_file, max_history,
                          self.interpreter, nlu_training_data,
                          should_merge_nodes, fontsize)

    def _ensure_agent_is_prepared(self):
        # type: () -> None
        """Checks that an interpreter and a tracker store are set.

        Necessary before a processor can be instantiated from this agent.
        Raises an exception if any argument is missing."""

        if self.interpreter is None or self.tracker_store is None:
            raise Exception("Agent needs to be prepared before usage. "
                            "You need to set an interpreter as well "
                            "as a tracker store.")

    def _create_processor(self, preprocessor=None):
        # type: (Optional[Callable[[Text], Text]]) -> MessageProcessor
        """Instantiates a processor based on the set state of the agent."""

        self._ensure_agent_is_prepared()
        return MessageProcessor(
                self.interpreter, self.policy_ensemble, self.domain,
                self.tracker_store, message_preprocessor=preprocessor)

    @staticmethod
    def _create_domain(domain):
        # type: (Union[Domain, Text]) -> Domain

        if isinstance(domain, string_types):
            return TemplateDomain.load(domain)
        elif isinstance(domain, Domain):
            return domain
        else:
            raise ValueError(
                    "Invalid param `domain`. Expected a path to a domain "
                    "specification or a domain instance. But got "
                    "type '{}' with value '{}'".format(type(domain), domain))

    @staticmethod
    def create_tracker_store(store, domain):
        # type: (Optional[TrackerStore], Domain) -> TrackerStore
        if store is not None:
            store.domain = domain
            return store
        else:
            return InMemoryTrackerStore(domain)

    @staticmethod
    def _create_interpreter(
            interp  # type: Union[Text, NaturalLanguageInterpreter, None]
    ):
        # type: (...) -> NaturalLanguageInterpreter
        return NaturalLanguageInterpreter.create(interp)

    @staticmethod
    def _create_ensemble(policies):
        # type: (Union[List[Policy], PolicyEnsemble, None]) -> PolicyEnsemble
        if policies is None:
            return SimplePolicyEnsemble([])
        if isinstance(policies, list):
            return SimplePolicyEnsemble(policies)
        elif isinstance(policies, PolicyEnsemble):
            return policies
        else:
            passed_type = type(policies).__name__
            raise ValueError(
                    "Invalid param `policies`. Passed object is "
                    "of type '{}', but should be policy, an array of "
                    "policies, or a policy ensemble".format(passed_type))
