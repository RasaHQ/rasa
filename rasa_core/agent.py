from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from threading import Thread

import six
import typing
from gevent.pywsgi import WSGIServer
from requests.exceptions import InvalidURL, RequestException
from six import string_types
from typing import Text, List, Optional, Callable, Any, Dict, Union

import rasa_core
from rasa_core import training, constants
from rasa_core.channels import UserMessage, OutputChannel, InputChannel
from rasa_core.constants import DEFAULT_REQUEST_TIMEOUT
from rasa_core.dispatcher import Dispatcher
from rasa_core.domain import Domain, check_domain_sanity
from rasa_core.exceptions import AgentNotReady
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.nlg import NaturalLanguageGenerator
from rasa_core.policies import Policy
from rasa_core.policies.ensemble import SimplePolicyEnsemble, PolicyEnsemble
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.processor import MessageProcessor
from rasa_core.tracker_store import InMemoryTrackerStore, TrackerStore
from rasa_core.trackers import DialogueStateTracker, EventVerbosity
from rasa_core.utils import EndpointConfig
from rasa_nlu.utils import is_url

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    # noinspection PyPep8Naming
    from rasa_core.nlg import NaturalLanguageGenerator as NLG

if six.PY2:
    # noinspection PyUnresolvedReferences
    from StringIO import StringIO as IOReader
else:
    from io import BytesIO as IOReader


def load_from_server(interpreter=None,  # type: NaturalLanguageInterpreter
                     generator=None,  # type: Union[EndpointConfig, NLG]
                     tracker_store=None,  # type: Optional[TrackerStore]
                     action_endpoint=None,  # type: Optional[EndpointConfig]
                     model_server=None,  # type: Optional[EndpointConfig]
                     wait_time_between_pulls=None,  # type: Optional[int]
                     ):
    # type: (...) -> Agent
    """Load a persisted model from a server."""

    agent = Agent(interpreter=interpreter,
                  generator=generator,
                  tracker_store=tracker_store,
                  action_endpoint=action_endpoint)

    if wait_time_between_pulls:
        # continuously pull the model every `wait_time_between_pulls` seconds
        start_model_pulling_in_worker(model_server,
                                      wait_time_between_pulls,
                                      agent)
    else:
        # just pull the model once
        _update_model_from_server(model_server, agent)

    return agent


def _init_model_from_server(model_server):
    # type: (EndpointConfig) -> Optional[(Domain, PolicyEnsemble, Text)]
    """Initialise a Rasa Core model from a URL."""

    if not is_url(model_server.url):
        raise InvalidURL(model_server.url)

    model_directory = tempfile.mkdtemp()

    fingerprint = _pull_model_and_fingerprint(model_server,
                                              model_directory,
                                              fingerprint=None)

    return fingerprint, model_directory


def _update_model_from_server(
        model_server,  # type: EndpointConfig
        agent,  # type: Agent
):
    # type: (...) -> None
    """Load a zipped Rasa Core model from a URL and update the passed agent."""

    if not is_url(model_server.url):
        raise InvalidURL(model_server.url)

    model_directory = tempfile.mkdtemp()

    new_model_fingerprint = _pull_model_and_fingerprint(
            model_server, model_directory, agent.fingerprint)
    if new_model_fingerprint:
        domain_path = os.path.join(os.path.abspath(model_directory),
                                   "domain.yml")
        domain = Domain.load(domain_path)
        policy_ensemble = PolicyEnsemble.load(model_directory)
        agent.update_model(domain, policy_ensemble, new_model_fingerprint)
    else:
        logger.debug("No new model found at "
                     "URL {}".format(model_server.url))


def _pull_model_and_fingerprint(model_server, model_directory, fingerprint):
    # type: (EndpointConfig, Text, Optional[Text]) -> Optional[Text]
    """Queries the model server and returns the value of the response's

    <ETag> header which contains the model hash."""
    header = {"If-None-Match": fingerprint}
    try:
        logger.debug("Requesting model from server {}..."
                     "".format(model_server.url))
        response = model_server.request(method="GET",
                                        headers=header,
                                        timeout=DEFAULT_REQUEST_TIMEOUT)
    except RequestException as e:
        logger.warning("Tried to fetch model from server, but couldn't reach "
                       "server. We'll retry later... Error: {}."
                       "".format(e))
        return None

    if response.status_code == 204:
        logger.debug("Model server returned 204 status code, indicating "
                     "that no new model is available. "
                     "Current fingerprint: {}".format(fingerprint))
        return response.headers.get("ETag")
    elif response.status_code == 404:
        logger.debug("Model server didn't find a model for our request. "
                     "Probably no one did train a model for the project "
                     "and tag combination yet.")
        return None
    elif response.status_code != 200:
        logger.warning("Tried to fetch model from server, but server response "
                       "status code is {}. We'll retry later..."
                       "".format(response.status_code))
        return None

    zip_ref = zipfile.ZipFile(IOReader(response.content))
    zip_ref.extractall(model_directory)
    logger.debug("Unzipped model to {}"
                 "".format(os.path.abspath(model_directory)))

    # get the new fingerprint
    return response.headers.get("ETag")


def _run_model_pulling_worker(model_server, wait_time_between_pulls, agent):
    # type: (EndpointConfig, int, Agent) -> None
    while True:
        _update_model_from_server(model_server, agent)
        time.sleep(wait_time_between_pulls)


def start_model_pulling_in_worker(model_server, wait_time_between_pulls, agent):
    # type: (EndpointConfig, int, Agent) -> None

    worker = Thread(target=_run_model_pulling_worker,
                    args=(model_server, wait_time_between_pulls, agent))
    worker.setDaemon(True)
    worker.start()


class Agent(object):
    """The Agent class provides a convenient interface for the most important
     Rasa Core functionality.

     This includes training, handling messages, loading a dialogue model,
     getting the next action, and handling a channel."""

    def __init__(
            self,
            domain=None,  # type: Union[Text, Domain]
            policies=None,  # type: Union[PolicyEnsemble, List[Policy], None]
            interpreter=None,  # type: Optional[NaturalLanguageInterpreter]
            generator=None,  # type: Union[EndpointConfig, NLG, None]
            tracker_store=None,  # type: Optional[TrackerStore]
            action_endpoint=None,  # type: Optional[EndpointConfig]
            fingerprint=None  # type: Optional[Text]
    ):
        # Initializing variables with the passed parameters.
        self.domain = self._create_domain(domain)
        self.policy_ensemble = self._create_ensemble(policies)

        if not isinstance(interpreter, NaturalLanguageInterpreter):
            if interpreter is not None:
                logger.warning(
                        "Passing a value for interpreter to an agent "
                        "where the value is not an interpreter "
                        "is deprecated. Construct the interpreter, before"
                        "passing it to the agent, e.g. "
                        "`interpreter = NaturalLanguageInterpreter.create("
                        "nlu)`.")
            interpreter = NaturalLanguageInterpreter.create(interpreter, None)

        self.interpreter = interpreter

        self.nlg = NaturalLanguageGenerator.create(generator, self.domain)
        self.tracker_store = self.create_tracker_store(
                tracker_store, self.domain)
        self.action_endpoint = action_endpoint

        self._set_fingerprint(fingerprint)

    def update_model(
            self,
            domain,  # type: Union[Text, Domain]
            policy_ensemble,  # type: PolicyEnsemble
            fingerprint  # type: Optional[Text]
    ):
        self.domain = domain
        self.policy_ensemble = policy_ensemble

        self._set_fingerprint(fingerprint)

        # update domain on all instances
        self.tracker_store.domain = domain
        if hasattr(self.nlg, "templates"):
            self.nlg.templates = domain.templates or []

    @classmethod
    def load(cls,
             path,  # type: Text
             interpreter=None,  # type: Optional[NaturalLanguageInterpreter]
             generator=None,  # type: Union[EndpointConfig, NLG]
             tracker_store=None,  # type: Optional[TrackerStore]
             action_endpoint=None,  # type: Optional[EndpointConfig]
             ):
        # type: (...) -> Agent
        """Load a persisted model from the passed path."""

        if not path:
            raise ValueError("You need to provide a valid directory where "
                             "to load the agent from when calling "
                             "`Agent.load`.")

        if os.path.isfile(path):
            raise ValueError("You are trying to load a MODEL from a file "
                             "('{}'), which is not possible. \n"
                             "The persisted path should be a directory "
                             "containing the various model files. \n\n"
                             "If you want to load training data instead of "
                             "a model, use `agent.load_data(...)` "
                             "instead.".format(path))

        domain = Domain.load(os.path.join(path, "domain.yml"))
        ensemble = PolicyEnsemble.load(path) if path else None

        # ensures the domain hasn't changed between test and train
        domain.compare_with_specification(path)

        return cls(domain=domain,
                   policies=ensemble,
                   interpreter=interpreter,
                   generator=generator,
                   tracker_store=tracker_store,
                   action_endpoint=action_endpoint)

    def is_ready(self):
        """Check if all necessary components are instantiated to use agent."""
        return (self.interpreter is not None and
                self.tracker_store is not None and
                self.policy_ensemble is not None)

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

        def noop(_):
            logger.info("Ignoring message as there is no agent to handle it.")
            return None

        if not self.is_ready():
            return noop(message)  #

        processor = self.create_processor(message_preprocessor)
        return processor.handle_message(message)

    # noinspection PyUnusedLocal
    def predict_next(
            self,
            sender_id,
            **kwargs
    ):
        # type: (Text, Any) -> Dict[Text, Any]
        """Handle a single message."""

        processor = self.create_processor()
        return processor.predict_next(sender_id)

    # noinspection PyUnusedLocal
    def log_message(
            self,
            message,  # type: UserMessage
            message_preprocessor=None,  # type: Optional[Callable[[Text], Text]]
            **kwargs  # type: Any
    ):
        # type: (...) -> DialogueStateTracker
        """Append a message to a dialogue - does not predict actions."""

        processor = self.create_processor(message_preprocessor)
        return processor.log_message(message)

    def execute_action(
            self,
            sender_id,  # type: Text
            action,  # type: Text
            output_channel  # type: OutputChannel
    ):
        # type: (...) -> DialogueStateTracker
        """Handle a single message."""

        processor = self.create_processor()
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
            >>> from rasa_core.interpreter import RasaNLUInterpreter
            >>> interpreter = RasaNLUInterpreter(
            ... "examples/restaurantbot/models/nlu/current")
            >>> agent = Agent.load("examples/restaurantbot/models/dialogue",
            ... interpreter=interpreter)
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
        from a different policy (e.g. ``KerasPolicy``). Useful to test
        prediction
        capabilities of an ensemble when ignoring memorized turns from the
        training data."""

        if not self.policy_ensemble:
            return

        for p in self.policy_ensemble.policies:
            # explicitly ignore inheritance (e.g. augmented memoization policy)
            if type(p) == MemoizationPolicy:
                p.toggle(activate)

    def continue_training(self,
                          trackers,
                          **kwargs
                          ):
        # type: (List[DialogueStateTracker], Any) -> None

        if not self.is_ready():
            raise AgentNotReady("Can't continue training without a policy "
                                "ensemble.")

        self.policy_ensemble.continue_training(trackers,
                                               self.domain,
                                               **kwargs)
        self._set_fingerprint()

    def load_data(self,
                  resource_name,  # type: Text
                  remove_duplicates=True,  # type: bool
                  unique_last_num_states=None,  # type: Optional[int]
                  augmentation_factor=20,  # type: int
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
                                  augmentation_factor,
                                  tracker_limit, use_story_concatenation,
                                  debug_plots)

    def train(self,
              training_trackers,  # type: List[DialogueStateTracker]
              **kwargs  # type: Any
              ):
        # type: (...) -> None
        """Train the policies / policy ensemble using dialogue data from file.

            :param training_trackers: trackers to train on
            :param kwargs: additional arguments passed to the underlying ML
                           trainer (e.g. keras parameters)
        """

        if not self.is_ready():
            raise AgentNotReady("Can't train without a policy ensemble.")

        # deprecation tests
        if kwargs.get('featurizer') or kwargs.get('max_history'):
            raise Exception("Passing `featurizer` and `max_history` "
                            "to `agent.train(...)` is not supported anymore. "
                            "Pass appropriate featurizer "
                            "directly to the policy instead. More info "
                            "https://rasa.com/docs/core/migrations.html#x-to"
                            "-0-9-0")

        if isinstance(training_trackers, string_types):
            # the user most likely passed in a file name to load training
            # data from
            raise Exception("Passing a file name to `agent.train(...)` is "
                            "not supported anymore. Rather load the data with "
                            "`data = agent.load_data(file_name)` and pass it "
                            "to `agent.train(data)`.")

        logger.debug("Agent trainer got kwargs: {}".format(kwargs))
        check_domain_sanity(self.domain)

        self.policy_ensemble.train(training_trackers, self.domain,
                                   **kwargs)
        self._set_fingerprint()

    def handle_channels(self, channels,
                        http_port=constants.DEFAULT_SERVER_PORT,
                        serve_forever=True):
        # type: (List[InputChannel], int, bool) -> WSGIServer
        """Start a webserver attaching the input channels and handling msgs.

        If ``serve_forever`` is set to ``True``, this call will be blocking.
        Otherwise the webserver will be started, and the method will
        return afterwards."""
        from flask import Flask

        app = Flask(__name__)
        rasa_core.channels.channel.register(channels,
                                            app,
                                            self.handle_message,
                                            route="/webhooks/")

        http_server = WSGIServer(('0.0.0.0', http_port), app)
        http_server.start()

        if serve_forever:
            http_server.serve_forever()
        return http_server

    def _set_fingerprint(self, fingerprint=None):
        # type: (Optional[Text]) -> None

        if fingerprint:
            self.fingerprint = fingerprint
        else:
            self.fingerprint = uuid.uuid4().hex

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
        # type: (Text, bool) -> None
        """Persists this agent into a directory for later loading and usage."""

        if not self.is_ready():
            raise AgentNotReady("Can't persist without a policy ensemble.")

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

    def _ensure_agent_is_ready(self):
        # type: () -> None
        """Checks that an interpreter and a tracker store are set.

        Necessary before a processor can be instantiated from this agent.
        Raises an exception if any argument is missing."""

        if not self.is_ready():
            raise AgentNotReady("Agent needs to be prepared before usage. "
                                "You need to set an interpreter, a policy "
                                "ensemble as well as a tracker store.")

    def create_processor(self, preprocessor=None):
        # type: (Optional[Callable[[Text], Text]]) -> MessageProcessor
        """Instantiates a processor based on the set state of the agent."""
        # Checks that the interpreter and tracker store are set and
        # creates a processor
        self._ensure_agent_is_ready()
        return MessageProcessor(
                self.interpreter,
                self.policy_ensemble,
                self.domain,
                self.tracker_store,
                self.nlg,
                action_endpoint=self.action_endpoint,
                message_preprocessor=preprocessor)

    @staticmethod
    def _create_domain(domain):
        # type: (Union[None, Domain, Text]) -> Domain

        if isinstance(domain, string_types):
            return Domain.load(domain)
        elif isinstance(domain, Domain):
            return domain
        elif domain is not None:
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
    def _create_ensemble(
            policies  # type: Union[List[Policy], PolicyEnsemble, None]
    ):
        # type: (...) -> Optional[PolicyEnsemble]
        if policies is None:
            return None
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
