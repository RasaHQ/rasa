from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from builtins import str
from typing import Text, List, Optional, Callable, Any, Dict

from rasa_core.channels import UserMessage, InputChannel, OutputChannel
from rasa_core.domain import TemplateDomain, Domain
from rasa_core.events import Event
from rasa_core.featurizers import Featurizer, BinaryFeaturizer
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.policies import PolicyTrainer
from rasa_core.policies.ensemble import SimplePolicyEnsemble, PolicyEnsemble
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.policies.online_policy_trainer import OnlinePolicyTrainer, \
    TrainingFinishedException
from rasa_core.processor import MessageProcessor
from rasa_core.tracker_store import InMemoryTrackerStore, TrackerStore


class Agent(object):
    """Public interface for common things to do.

     This includes e.g. train an assistant, or handle messages
     with an assistant."""

    def __init__(self,
                 domain,
                 policies=None,
                 featurizer=None,
                 interpreter=None,
                 tracker_store=None):
        self.domain = self._create_domain(domain)
        self.featurizer = self._create_featurizer(featurizer)
        self.policy_ensemble = self._create_ensemble(policies)
        self.interpreter = NaturalLanguageInterpreter.create(interpreter)
        self.tracker_store = self._create_tracker_store(
                tracker_store, self.domain)

    @classmethod
    def load(cls, path, interpreter=None, tracker_store=None):
        # type: (Text, Any, Optional[TrackerStore]) -> Agent

        if path is None:
            raise ValueError("No domain path specified.")
        domain = TemplateDomain.load(os.path.join(path, "domain.yml"))
        # ensures the domain hasn't changed between test and train
        domain.compare_with_specification(path)
        featurizer = Featurizer.load(path)
        ensemble = PolicyEnsemble.load(path, featurizer)
        _interpreter = NaturalLanguageInterpreter.create(interpreter)
        _tracker_store = cls._create_tracker_store(tracker_store, domain)
        return cls(domain, ensemble, featurizer, _interpreter, _tracker_store)

    def handle_message(
            self,
            text_message,  # type: Text
            message_preprocessor=None,  # type: Optional[Callable[[Text], Text]]
            output_channel=None,  # type: Optional[OutputChannel]
            sender=None  # type: Optional[Text]
    ):
        # type: (...) -> Optional[List[Text]]
        """



        Handle a single message.

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
                UserMessage(text_message, output_channel, sender))

    def start_message_handling(self,
                               text_message,
                               sender=None):
        # type: (Text, Optional[Text]) -> Dict[Text, Any]

        processor = self._create_processor()
        return processor.start_message_handling(
                UserMessage(text_message, None, sender))

    def continue_message_handling(self, sender_id, executed_action, events):
        # type: (Text, Text, List[Event]) -> Dict[Text, Any]

        processor = self._create_processor()
        return processor.continue_message_handling(sender_id,
                                                   executed_action,
                                                   events)

    def handle_channel(self, input_channel,
                       message_preprocessor=None):
        # type: (InputChannel, Optional[Callable[[Text], Text]]) -> None
        """Handle messages coming from the channel."""

        processor = self._create_processor(message_preprocessor)
        processor.handle_channel(input_channel)

    def toggle_memoization(self, activate):
        # type: (bool) -> None
        """Toggles the memoization on and off.

        If a memoization policy is present in the ensemble, this will toggle
        the prediction of that policy. When set to `false` the Memoization
        policies present in the policy ensemble will not make any predictions.
        Hence, the prediction result from the ensemble always needs to come
        from a different policy (e.g. `KerasPolicy`). Useful to test prediction
        capabilities of an ensemble when ignoring memorized turns from the
        training data."""

        for p in self.policy_ensemble.policies:
            # explicitly ignore inheritance (e.g. scoring policy)
            if type(p) == MemoizationPolicy:
                p.toggle(activate)

    def train(self, filename=None, model_path=None, remove_duplicates=True, **kwargs):
        # type: (Optional[Text], Optional[Text], **Any) -> None
        """Train the policies / policy ensemble using dialogue data from file"""

        trainer = PolicyTrainer(self.policy_ensemble, self.domain,
                                self.featurizer)
        trainer.train(filename, **kwargs)

        if model_path:
            self.persist(model_path)

    def train_online(self,
                     filename=None,  # type: Optional[Text]
                     input_channel=None,  # type: Optional[InputChannel]
                     model_path=None,  # type: Optional[Text]
                     **kwargs  # type: **Any
                     ):
        # type: (...) -> None
        """Runs an online training session on the set policies / ensemble.

        The policies will be pretrained using the data from `filename`.
        After that the model will get trained on dialogues from the input
        channel. During the dialogue the annotations and state of the agent
        can be changed to correct wrong behaviour."""

        if not self.interpreter:
            raise ValueError(
                    "When using online learning, you need to specify "
                    "an interpreter for the agent to use.")
        trainer = OnlinePolicyTrainer(self.policy_ensemble, self.domain,
                                      self.featurizer)
        trainer.train(filename, self.interpreter, input_channel, **kwargs)

        if model_path:
            self.persist(model_path)

    def persist(self, model_path):
        # type: (Text) -> None
        """Persists this agent into a directory for later loading and usage."""

        self.policy_ensemble.persist(model_path)
        self.domain.persist(os.path.join(model_path, "domain.yml"))
        self.domain.persist_specification(model_path)
        self.featurizer.persist(model_path)

    def _ensure_agent_is_prepared(self):
        # type: () -> None
        """Checks that an interpreter and a tracker store are set.

        Necessary before a processor can be instantiated from this agent.
        Raises an exception if any argument is missing."""

        if self.interpreter is None or self.tracker_store is None:
            raise Exception(
                    "Agent needs to be prepared before usage. "
                    "You need to set an interpreter as well "
                    "as a tracker store.")

    def _create_processor(self, preprocessor=None):
        # type: (Callable[[Text], Text]) -> MessageProcessor
        """Instantiates a processor based on the set state of the agent."""

        self._ensure_agent_is_prepared()
        return MessageProcessor(
                self.interpreter, self.policy_ensemble, self.domain,
                self.tracker_store, message_preprocessor=preprocessor)

    @classmethod
    def _create_featurizer(cls, featurizer):
        return featurizer if featurizer is not None else BinaryFeaturizer()

    @classmethod
    def _create_domain(cls, domain):
        if isinstance(domain, str):
            return TemplateDomain.load(domain)
        elif isinstance(domain, Domain):
            return domain
        else:
            raise ValueError(
                    "Invalid param `domain`. Expected a path to a domain "
                    "specification or a domain instance. But got "
                    "type '{}' with value '{}'".format(type(domain), domain))

    @classmethod
    def _create_tracker_store(cls, store, domain):
        return store if store is not None else InMemoryTrackerStore(domain)

    @staticmethod
    def _create_interpreter(interp):
        return NaturalLanguageInterpreter.create(interp)

    @staticmethod
    def _create_ensemble(policies):
        if policies is None:
            return SimplePolicyEnsemble([MemoizationPolicy])
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
