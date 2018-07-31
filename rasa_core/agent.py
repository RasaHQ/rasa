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
from rasa_core.dispatcher import Dispatcher
from rasa_core.utils import EndpointConfig
from rasa_core.channels import UserMessage, InputChannel, OutputChannel
from rasa_core.domain import TemplateDomain, Domain, check_domain_sanity
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.nlg import NaturalLanguageGenerator
from rasa_core.policies import Policy
from rasa_core.policies.ensemble import SimplePolicyEnsemble, PolicyEnsemble
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.processor import MessageProcessor
from rasa_core.tracker_store import InMemoryTrackerStore, TrackerStore
from rasa_core.trackers import DialogueStateTracker
from rasa_core.utils import EndpointConfig

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.interpreter import NaturalLanguageInterpreter as NLI
    from rasa_core.nlg import NaturalLanguageGenerator as NLG


class Agent(object):
    """The Agent class provides a convenient interface for the most important
     Rasa Core functionality.

     This includes training, handling messages, loading a dialogue model,
     getting the next action, and handling a channel."""

    def __init__(
            self,
            domain,  # type: Union[Text, Domain]
            policies=None,  # type: Union[PolicyEnsemble, List[Policy], None]
            interpreter=None,  # type: Union[NLI, Text, None]
            generator=None,  # type: Union[EndpointConfig, NLG]
            tracker_store=None,  # type: Optional[TrackerStore]
            action_endpoint=None,  # type: Optional[EndpointConfig]
    ):
        # Initializing variables with the passed parameters.
        self.domain = self._create_domain(domain, action_endpoint)
        self.policy_ensemble = self._create_ensemble(policies)
        self.interpreter = NaturalLanguageInterpreter.create(interpreter)
        self.nlg = NaturalLanguageGenerator.create(generator, self.domain)
        self.tracker_store = self.create_tracker_store(
                tracker_store, self.domain)

    @classmethod
    def load(cls,
             path,  # type: Text
             interpreter=None,  # type: Union[NLI, Text, None]
             generator=None,  # type: Union[EndpointConfig, NLG]
             tracker_store=None,  # type: Optional[TrackerStore]
             action_endpoint=None,  # type: Optional[EndpointConfig]
             ):
        # type: (...) -> Agent
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
                                     action_endpoint)
        # ensures the domain hasn't changed between test and train
        domain.compare_with_specification(path)

        return cls(domain, ensemble, interpreter,
                   generator, tracker_store, action_endpoint)

    def handle_message(
            self,
            message,  # type: UserMessage
            message_preprocessor=None,  # type: Optional[Callable[[Text], Text]]
            **kwargs
    ):
        # type: (...) -> Optional[List[Text]]
        """Handle a single message."""

        if not isinstance(message, UserMessage):
            logger.warning("Passing a text to `agent.handle_message(...)` is "
                           "deprecated. Rather use `agent.handle_text(...)`.")
            return self.handle_text(message,
                                    message_preprocessor=message_preprocessor,
                                    **kwargs)

        processor = self._create_processor(message_preprocessor)
        return processor.handle_message(message)

    def predict_next(
            self,
            sender_id,
            **kwargs
    ):
        # type: (Text, **Any) -> Dict[Text, Any]
        """Handle a single message."""

        processor = self._create_processor()
        return processor.predict_next(sender_id)

    def log_message(
            self,
            message,  # type: UserMessage
            message_preprocessor=None,  # type: Optional[Callable[[Text], Text]]
            **kwargs
    ):
        # type: (...) -> Dict[Text, Any]
        """Handle a single message."""

        processor = self._create_processor(message_preprocessor)
        return processor.log_message(message)

    def execute_action(
            self,
            sender_id,
            action,  # type: Text
            output_channel
    ):
        # type: (...) -> DialogueStateTracker
        """Handle a single message."""

        processor = self._create_processor()
        dispatcher = Dispatcher(sender_id,
                                output_channel,
                                self.nlg)
        return processor.execute_action(sender_id, action, dispatcher)

    def handle_text(
            self,
            text_message,  # type: Union[Text, Dict[Text, Any]]
            message_preprocessor=None,  # type: Optional[Callable[[Text], Text]]
            output_channel=None,  # type: Optional[OutputChannel]
            sender_id=UserMessage.DEFAULT_SENDER_ID  # type: Optional[Text]
    ):
        # type: (...) -> Optional[List[Text]]
        """Handle a single message.

        If a message preprocessor is passed, the message will be passed to that
        function first and the return value is then used as the
        input for the dialogue engine.

        The return value of this function depends on the ``output_channel``. If
        the output channel is not set, set to ``None``, or set
        to ``CollectingOutputChannel`` this function will return the messages
        the bot wants to respond.

        :Example:

            >>> from rasa_core.agent import Agent
            >>> agent = Agent.load("examples/restaurantbot/models/dialogue",
            ... interpreter="examples/restaurantbot/models/nlu/current")
            >>> agent.handle_text("hello")
            [u'how can I help you?']

        """

        if isinstance(text_message, string_types):
            text_message = {"text": text_message}

        msg = UserMessage(text_message.get("text"),
                          output_channel,
                          sender_id)

        return self.handle_message(msg, message_preprocessor)

    def toggle_memoization(
            self,
            activate  # type: bool
    ):
        # type: (...) -> None
        """Toggles the memoization on and off.

        If a memoization policy is present in the ensemble, this will toggle
        the prediction of that policy. When set to ``False`` the Memoization
        policies present in the policy ensemble will not make any predictions.
        Hence, the prediction result from the ensemble always needs to come
        from a different policy (e.g. ``KerasPolicy``). Useful to test prediction
        capabilities of an ensemble when ignoring memorized turns from the
        training data."""

        for p in self.policy_ensemble.policies:
            # explicitly ignore inheritance (e.g. augmented memoization policy)
            if type(p) == MemoizationPolicy:
                p.toggle(activate)

    def continue_training(self,
                          trackers,
                          **kwargs
                          ):
        # type: (List[DialogueStateTracker], **Any) -> None
        self.policy_ensemble.continue_training(trackers,
                                               self.domain,
                                               **kwargs)

    def load_data(self,
                  resource_name,  # type: Text
                  remove_duplicates=True,  # type: bool
                  unique_last_num_states=None,  # type: Optional[int]
                  augmentation_factor=20,  # type: int
                  max_number_of_trackers=None,  # deprecated
                  tracker_limit=None,  # type: Optional[int]
                  use_story_concatenation=True,  # type: bool
                  debug_plots=False  # type: bool
                  ):
        # type: (...) -> List[DialogueStateTracker]
        """Load training data from a resource."""

        # find maximum max_history
        # and if all featurizers are MaxHistoryTrackerFeaturizer
        max_max_history = 0
        all_max_history_featurizers = True
        for policy in self.policy_ensemble.policies:
            if hasattr(policy.featurizer, 'max_history'):
                max_max_history = max(policy.featurizer.max_history,
                                      max_max_history)
            elif policy.featurizer is not None:
                all_max_history_featurizers = False

        if unique_last_num_states is None:
            # for speed up of data generation
            # automatically detect unique_last_num_states
            # if it was not set and
            # if all featurizers are MaxHistoryTrackerFeaturizer
            if all_max_history_featurizers:
                unique_last_num_states = max_max_history
        elif unique_last_num_states < max_max_history:
            # possibility of data loss
            logger.warning("unique_last_num_states={} but "
                           "maximum max_history={}."
                           "Possibility of data loss. "
                           "It is recommended to set "
                           "unique_last_num_states to "
                           "at least maximum max_history."
                           "".format(unique_last_num_states, max_max_history))

        return training.load_data(resource_name, self.domain,
                                  remove_duplicates, unique_last_num_states,
                                  augmentation_factor, max_number_of_trackers,
                                  tracker_limit, use_story_concatenation,
                                  debug_plots)

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

    def persist(self, model_path, dump_flattened_stories=False):
        # type: (Text) -> None
        """Persists this agent into a directory for later loading and usage."""

        self._clear_model_directory(model_path)

        self.policy_ensemble.persist(model_path, dump_flattened_stories)
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
        # Checks that the interpreter and tracker store are set and
        # creates a processor
        self._ensure_agent_is_prepared()
        return MessageProcessor(
                self.interpreter, self.policy_ensemble, self.domain,
                self.tracker_store, self.nlg, message_preprocessor=preprocessor)

    @staticmethod
    def _create_domain(domain, action_endpoint=None):
        # type: (Union[Domain, Text]) -> Domain

        if isinstance(domain, string_types):
            return TemplateDomain.load(domain, action_endpoint)
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
            interp  # type: Union[Text, NLI, None]
    ):
        # type: (...) -> NLI
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
