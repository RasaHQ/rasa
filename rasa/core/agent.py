import asyncio
import logging
import os
import shutil
import tempfile
import typing
import uuid
from asyncio import CancelledError
from typing import Any, Callable, Dict, List, Optional, Text, Union

import aiohttp

from rasa.core import constants, jobs, training, utils
from rasa.core.channels import InputChannel, OutputChannel, UserMessage
from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT
from rasa.core.dispatcher import Dispatcher
from rasa.core.domain import Domain, InvalidDomain, check_domain_sanity
from rasa.core.exceptions import AgentNotReady
from rasa.core.interpreter import NaturalLanguageInterpreter
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.core.policies import FormPolicy, Policy
from rasa.core.policies.ensemble import PolicyEnsemble, SimplePolicyEnsemble
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.processor import MessageProcessor
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.core.trackers import DialogueStateTracker
from rasa.core.utils import LockCounter
from rasa.nlu.utils import is_url
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    # noinspection PyPep8Naming
    from rasa.core.nlg import NaturalLanguageGenerator as NLG
    from rasa.core.tracker_store import TrackerStore
    from sanic import Sanic


async def load_from_server(
    agent, model_server: Optional[EndpointConfig] = None
) -> "Agent":
    """Load a persisted model from a server."""

    # We are going to pull the model once first, and then schedule a recurring
    # job. the benefit of this approach is that we can be sure that there
    # is a model after this function completes -> allows to do proper
    # "is alive" check on a startup server's `/status` endpoint. If the server
    # is started, we can be sure that it also already loaded (or tried to)
    # a model.
    await _update_model_from_server(model_server, agent)

    wait_time_between_pulls = model_server.kwargs.get("wait_time_between_pulls", 100)

    if wait_time_between_pulls:
        # continuously pull the model every `wait_time_between_pulls` seconds
        await schedule_model_pulling(model_server, int(wait_time_between_pulls), agent)

    return agent


def _get_stack_model_directory(model_directory: Text) -> Optional[Text]:
    """Decide whether a persisted model is a stack or a core model.

    Return the root stack model directory if it's a stack model.
    """

    for root, _, files in os.walk(model_directory):
        if "fingerprint.json" in files:
            return root


def _load_and_set_updated_model(
    agent: "Agent", model_directory: Text, fingerprint: Text
):
    """Load the persisted model into memory and set the model on the agent."""

    logger.debug("Found new model with fingerprint {}. Loading...".format(fingerprint))

    stack_model_directory = _get_stack_model_directory(model_directory)
    if stack_model_directory:
        from rasa.core.interpreter import RasaNLUInterpreter

        nlu_model = os.path.join(stack_model_directory, "nlu")
        core_model = os.path.join(stack_model_directory, "core")
        interpreter = RasaNLUInterpreter(model_directory=nlu_model)
    else:
        interpreter = agent.interpreter
        core_model = model_directory

    domain_path = os.path.join(os.path.abspath(core_model), "domain.yml")
    domain = Domain.load(domain_path)

    # noinspection PyBroadException
    try:
        policy_ensemble = PolicyEnsemble.load(core_model)
        agent.update_model(domain, policy_ensemble, fingerprint, interpreter)
        logger.debug("Finished updating agent to new model.")
    except Exception:
        logger.exception(
            "Failed to load policy and update agent. "
            "The previous model will stay loaded instead."
        )


async def _update_model_from_server(
    model_server: EndpointConfig, agent: "Agent"
) -> None:
    """Load a zipped Rasa Core model from a URL and update the passed agent."""

    if not is_url(model_server.url):
        raise aiohttp.InvalidURL(model_server.url)

    model_directory = tempfile.mkdtemp()

    new_model_fingerprint = await _pull_model_and_fingerprint(
        model_server, model_directory, agent.fingerprint
    )
    if new_model_fingerprint:
        _load_and_set_updated_model(agent, model_directory, new_model_fingerprint)
    else:
        logger.debug("No new model found at URL {}".format(model_server.url))


async def _pull_model_and_fingerprint(
    model_server: EndpointConfig, model_directory: Text, fingerprint: Optional[Text]
) -> Optional[Text]:
    """Queries the model server and returns the value of the response's

     <ETag> header which contains the model hash.
     """

    headers = {"If-None-Match": fingerprint}

    logger.debug("Requesting model from server {}...".format(model_server.url))

    async with model_server.session() as session:
        try:
            params = model_server.combine_parameters()
            async with session.request(
                "GET",
                model_server.url,
                timeout=DEFAULT_REQUEST_TIMEOUT,
                headers=headers,
                params=params,
            ) as resp:

                if resp.status in [204, 304]:
                    logger.debug(
                        "Model server returned {} status code, "
                        "indicating that no new model is available. "
                        "Current fingerprint: {}"
                        "".format(resp.status, fingerprint)
                    )
                    return resp.headers.get("ETag")
                elif resp.status == 404:
                    logger.debug(
                        "Model server didn't find a model for our request. "
                        "Probably no one did train a model for the project "
                        "and tag combination yet."
                    )
                    return None
                elif resp.status != 200:
                    logger.warning(
                        "Tried to fetch model from server, but server response "
                        "status code is {}. We'll retry later..."
                        "".format(resp.status)
                    )
                    return None

                utils.unarchive(await resp.read(), model_directory)
                logger.debug(
                    "Unzipped model to '{}'".format(os.path.abspath(model_directory))
                )

                # get the new fingerprint
                return resp.headers.get("ETag")

        except aiohttp.ClientError as e:
            logger.info(
                "Tried to fetch model from server, but "
                "couldn't reach server. We'll retry later... "
                "Error: {}.".format(e)
            )

            return None


async def _run_model_pulling_worker(
    model_server: EndpointConfig, wait_time_between_pulls: int, agent: "Agent"
) -> None:
    # noinspection PyBroadException
    try:
        await _update_model_from_server(model_server, agent)
    except CancelledError:
        logger.warning("Stopping model pulling (cancelled).")
    except Exception:
        logger.exception(
            "An exception was raised while fetching a model. Continuing anyways..."
        )


async def schedule_model_pulling(
    model_server: EndpointConfig, wait_time_between_pulls: int, agent: "Agent"
):
    (await jobs.scheduler()).add_job(
        _run_model_pulling_worker,
        "interval",
        seconds=wait_time_between_pulls,
        args=[model_server, wait_time_between_pulls, agent],
        id="pull-model-from-server",
        replace_existing=True,
    )


class Agent(object):
    """The Agent class provides a convenient interface for the most important
     Rasa Core functionality.

     This includes training, handling messages, loading a dialogue model,
     getting the next action, and handling a channel."""

    def __init__(
        self,
        domain: Union[Text, Domain] = None,
        policies: Union[PolicyEnsemble, List[Policy], None] = None,
        interpreter: Optional[NaturalLanguageInterpreter] = None,
        generator: Union[EndpointConfig, "NLG", None] = None,
        tracker_store: Optional["TrackerStore"] = None,
        action_endpoint: Optional[EndpointConfig] = None,
        fingerprint: Optional[Text] = None,
    ):
        # Initializing variables with the passed parameters.
        self.domain = self._create_domain(domain)
        if self.domain:
            self.domain.add_requested_slot()
        self.policy_ensemble = self._create_ensemble(policies)
        if not self._is_form_policy_present():
            raise InvalidDomain(
                "You have defined a form action, but haven't added the "
                "FormPolicy to your policy ensemble."
            )

        self.interpreter = NaturalLanguageInterpreter.create(interpreter)

        self.nlg = NaturalLanguageGenerator.create(generator, self.domain)
        self.tracker_store = self.create_tracker_store(tracker_store, self.domain)
        self.action_endpoint = action_endpoint
        self.conversations_in_processing = {}

        self._set_fingerprint(fingerprint)

    def update_model(
        self,
        domain: Union[Text, Domain],
        policy_ensemble: PolicyEnsemble,
        fingerprint: Optional[Text],
        interpreter: Optional[NaturalLanguageInterpreter] = None,
    ) -> None:
        self.domain = domain
        self.policy_ensemble = policy_ensemble

        if interpreter:
            self.interpreter = NaturalLanguageInterpreter.create(interpreter)

        self._set_fingerprint(fingerprint)

        # update domain on all instances
        self.tracker_store.domain = domain
        if hasattr(self.nlg, "templates"):
            self.nlg.templates = domain.templates or []

    @classmethod
    def load(
        cls,
        path: Text,
        interpreter: Optional[NaturalLanguageInterpreter] = None,
        generator: Union[EndpointConfig, "NLG"] = None,
        tracker_store: Optional["TrackerStore"] = None,
        action_endpoint: Optional[EndpointConfig] = None,
    ) -> "Agent":
        """Load a persisted model from the passed path."""

        if not path:
            raise ValueError(
                "You need to provide a valid directory where "
                "to load the agent from when calling "
                "`Agent.load`."
            )

        if os.path.isfile(path):
            raise ValueError(
                "You are trying to load a MODEL from a file "
                "('{}'), which is not possible. \n"
                "The persisted path should be a directory "
                "containing the various model files. \n\n"
                "If you want to load training data instead of "
                "a model, use `agent.load_data(...)` "
                "instead.".format(path)
            )

        domain = Domain.load(os.path.join(path, "domain.yml"))
        ensemble = PolicyEnsemble.load(path) if path else None

        # ensures the domain hasn't changed between test and train
        domain.compare_with_specification(path)

        return cls(
            domain=domain,
            policies=ensemble,
            interpreter=interpreter,
            generator=generator,
            tracker_store=tracker_store,
            action_endpoint=action_endpoint,
        )

    def is_ready(self):
        """Check if all necessary components are instantiated to use agent."""
        return (
            self.interpreter is not None
            and self.tracker_store is not None
            and self.policy_ensemble is not None
        )

    async def handle_message(
        self,
        message: UserMessage,
        message_preprocessor: Optional[Callable[[Text], Text]] = None,
        **kwargs
    ) -> Optional[List[Text]]:
        """Handle a single message."""

        if not isinstance(message, UserMessage):
            logger.warning(
                "Passing a text to `agent.handle_message(...)` is "
                "deprecated. Rather use `agent.handle_text(...)`."
            )
            # noinspection PyTypeChecker
            return await self.handle_text(
                message, message_preprocessor=message_preprocessor, **kwargs
            )

        def noop(_):
            logger.info("Ignoring message as there is no agent to handle it.")
            return None

        if not self.is_ready():
            return noop(message)

        processor = self.create_processor(message_preprocessor)

        # get the lock for the current conversation
        lock = self.conversations_in_processing.get(message.sender_id)
        if not lock:
            logger.debug(
                "created a new lock for conversation '{}'".format(message.sender_id)
            )
            lock = LockCounter()
            self.conversations_in_processing[message.sender_id] = lock

        try:
            async with lock:
                # this makes sure that there can always only be one coroutine
                # handling a conversation at any point in time
                # Note: this doesn't support multi-processing, it just works
                # for coroutines. If there are multiple processes handling
                # messages, an external system needs to make sure messages
                # for the same conversation are always processed by the same
                # process.
                return await processor.handle_message(message)
        finally:
            if not lock.is_someone_waiting():
                # dispose of the lock if no one needs it to avoid
                # accumulating locks
                del self.conversations_in_processing[message.sender_id]
                logger.debug(
                    "deleted lock for conversation '{}' (unused)"
                    "".format(message.sender_id)
                )

    # noinspection PyUnusedLocal
    def predict_next(self, sender_id: Text, **kwargs: Any) -> Dict[Text, Any]:
        """Handle a single message."""

        processor = self.create_processor()
        return processor.predict_next(sender_id)

    # noinspection PyUnusedLocal
    async def log_message(
        self,
        message: UserMessage,
        message_preprocessor: Optional[Callable[[Text], Text]] = None,
        **kwargs: Any
    ) -> DialogueStateTracker:
        """Append a message to a dialogue - does not predict actions."""

        processor = self.create_processor(message_preprocessor)
        return await processor.log_message(message)

    async def execute_action(
        self,
        sender_id: Text,
        action: Text,
        output_channel: OutputChannel,
        policy: Text,
        confidence: float,
    ) -> DialogueStateTracker:
        """Handle a single message."""

        processor = self.create_processor()
        dispatcher = Dispatcher(sender_id, output_channel, self.nlg)
        return await processor.execute_action(
            sender_id, action, dispatcher, policy, confidence
        )

    async def handle_text(
        self,
        text_message: Union[Text, Dict[Text, Any]],
        message_preprocessor: Optional[Callable[[Text], Text]] = None,
        output_channel: Optional[OutputChannel] = None,
        sender_id: Optional[Text] = UserMessage.DEFAULT_SENDER_ID,
    ) -> Optional[List[Dict[Text, Any]]]:
        """Handle a single message.

        If a message preprocessor is passed, the message will be passed to that
        function first and the return value is then used as the
        input for the dialogue engine.

        The return value of this function depends on the ``output_channel``. If
        the output channel is not set, set to ``None``, or set
        to ``CollectingOutputChannel`` this function will return the messages
        the bot wants to respond.

        :Example:

            >>> from rasa.core.agent import Agent
            >>> from rasa.core.interpreter import RasaNLUInterpreter
            >>> interpreter = RasaNLUInterpreter(
            ... "examples/restaurantbot/models/nlu/current")
            >>> agent = Agent.load("examples/restaurantbot/models/dialogue",
            ... interpreter=interpreter)
            >>> await agent.handle_text("hello")
            [u'how can I help you?']

        """

        if isinstance(text_message, str):
            text_message = {"text": text_message}

        msg = UserMessage(text_message.get("text"), output_channel, sender_id)

        return await self.handle_message(msg, message_preprocessor)

    def toggle_memoization(self, activate: bool) -> None:
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

    def continue_training(
        self, trackers: List[DialogueStateTracker], **kwargs: Any
    ) -> None:

        if not self.is_ready():
            raise AgentNotReady("Can't continue training without a policy ensemble.")

        self.policy_ensemble.continue_training(trackers, self.domain, **kwargs)
        self._set_fingerprint()

    def _max_history(self):
        """Find maximum max_history."""

        max_histories = [
            policy.featurizer.max_history
            for policy in self.policy_ensemble.policies
            if hasattr(policy.featurizer, "max_history")
        ]

        return max(max_histories or [0])

    def _are_all_featurizers_using_a_max_history(self):
        """Check if all featurizers are MaxHistoryTrackerFeaturizer."""

        def has_max_history_featurizer(policy):
            return policy.featurizer and hasattr(policy.featurizer, "max_history")

        for p in self.policy_ensemble.policies:
            if p.featurizer and not has_max_history_featurizer(p):
                return False
        return True

    async def load_data(
        self,
        resource_name: Text,
        remove_duplicates: bool = True,
        unique_last_num_states: Optional[int] = None,
        augmentation_factor: int = 20,
        tracker_limit: Optional[int] = None,
        use_story_concatenation: bool = True,
        debug_plots: bool = False,
        exclusion_percentage: int = None,
    ) -> List[DialogueStateTracker]:
        """Load training data from a resource."""

        max_history = self._max_history()

        if unique_last_num_states is None:
            # for speed up of data generation
            # automatically detect unique_last_num_states
            # if it was not set and
            # if all featurizers are MaxHistoryTrackerFeaturizer
            if self._are_all_featurizers_using_a_max_history():
                unique_last_num_states = max_history
        elif unique_last_num_states < max_history:
            # possibility of data loss
            logger.warning(
                "unique_last_num_states={} but "
                "maximum max_history={}."
                "Possibility of data loss. "
                "It is recommended to set "
                "unique_last_num_states to "
                "at least maximum max_history."
                "".format(unique_last_num_states, max_history)
            )

        return await training.load_data(
            resource_name,
            self.domain,
            remove_duplicates,
            unique_last_num_states,
            augmentation_factor,
            tracker_limit,
            use_story_concatenation,
            debug_plots,
            exclusion_percentage=exclusion_percentage,
        )

    def train(
        self, training_trackers: List[DialogueStateTracker], **kwargs: Any
    ) -> None:
        """Train the policies / policy ensemble using dialogue data from file.

        Args:
            training_trackers: trackers to train on
            **kwargs: additional arguments passed to the underlying ML
                           trainer (e.g. keras parameters)
        """
        if not self.is_ready():
            raise AgentNotReady("Can't train without a policy ensemble.")

        # deprecation tests
        if kwargs.get("featurizer"):
            raise Exception(
                "Passing `featurizer` "
                "to `agent.train(...)` is not supported anymore. "
                "Pass appropriate featurizer directly "
                "to the policy configuration instead. More info "
                "https://rasa.com/docs/core/migrations.html"
            )
        if (
            kwargs.get("epochs")
            or kwargs.get("max_history")
            or kwargs.get("batch_size")
        ):
            raise Exception(
                "Passing policy configuration parameters "
                "to `agent.train(...)` is not supported "
                "anymore. Specify parameters directly in the "
                "policy configuration instead. More info "
                "https://rasa.com/docs/core/migrations.html"
            )

        if isinstance(training_trackers, str):
            # the user most likely passed in a file name to load training
            # data from
            raise Exception(
                "Passing a file name to `agent.train(...)` is "
                "not supported anymore. Rather load the data with "
                "`data = agent.load_data(file_name)` and pass it "
                "to `agent.train(data)`."
            )

        logger.debug("Agent trainer got kwargs: {}".format(kwargs))

        check_domain_sanity(self.domain)

        self.policy_ensemble.train(training_trackers, self.domain, **kwargs)
        self._set_fingerprint()

    def handle_channels(
        self,
        channels: List[InputChannel],
        http_port: int = constants.DEFAULT_SERVER_PORT,
        route: Text = "/webhooks/",
        cors=None,
    ) -> "Sanic":
        """Start a webserver attaching the input channels and handling msgs.

        If ``serve_forever`` is set to ``True``, this call will be blocking.
        Otherwise the webserver will be started, and the method will
        return afterwards."""
        from rasa.core import run

        app = run.configure_app(channels, cors, None, enable_api=False, route=route)

        app.agent = self

        app.run(
            host="0.0.0.0",
            port=http_port,
            access_log=logger.isEnabledFor(logging.DEBUG),
        )

        # this might seem unnecessary (as run does not return until the server
        # is killed) - but we use it for tests where we mock `.run` to directly
        # return and need the app to inspect if we created a properly
        # configured server
        return app

    def _set_fingerprint(self, fingerprint: Optional[Text] = None) -> None:

        if fingerprint:
            self.fingerprint = fingerprint
        else:
            self.fingerprint = uuid.uuid4().hex

    @staticmethod
    def _clear_model_directory(model_path: Text) -> None:
        """Remove existing files from model directory.

        Only removes files if the directory seems to contain a previously
        persisted model. Otherwise does nothing to avoid deleting
        `/` by accident."""

        if not os.path.exists(model_path):
            return

        domain_spec_path = os.path.join(model_path, "metadata.json")
        # check if there were a model before
        if os.path.exists(domain_spec_path):
            logger.info(
                "Model directory {} exists and contains old "
                "model files. All files will be overwritten."
                "".format(model_path)
            )
            shutil.rmtree(model_path)
        else:
            logger.debug(
                "Model directory {} exists, but does not contain "
                "all old model files. Some files might be "
                "overwritten.".format(model_path)
            )

    def persist(self, model_path: Text, dump_flattened_stories: bool = False) -> None:
        """Persists this agent into a directory for later loading and usage."""

        if not self.is_ready():
            raise AgentNotReady("Can't persist without a policy ensemble.")

        self._clear_model_directory(model_path)

        self.policy_ensemble.persist(model_path, dump_flattened_stories)
        self.domain.persist(os.path.join(model_path, "domain.yml"))
        self.domain.persist_specification(model_path)

        logger.info("Persisted model to '{}'".format(os.path.abspath(model_path)))

    async def visualize(
        self,
        resource_name: Text,
        output_file: Text,
        max_history: Optional[int] = None,
        nlu_training_data: Optional[Text] = None,
        should_merge_nodes: bool = True,
        fontsize: int = 12,
    ) -> None:
        from rasa.core.training.visualization import visualize_stories
        from rasa.core.training.dsl import StoryFileReader

        """Visualize the loaded training data from the resource."""

        # if the user doesn't provide a max history, we will use the
        # largest value from any policy
        max_history = max_history or self._max_history()

        story_steps = await StoryFileReader.read_from_folder(resource_name, self.domain)
        await visualize_stories(
            story_steps,
            self.domain,
            output_file,
            max_history,
            self.interpreter,
            nlu_training_data,
            should_merge_nodes,
            fontsize,
        )

    def _ensure_agent_is_ready(self) -> None:
        """Checks that an interpreter and a tracker store are set.

        Necessary before a processor can be instantiated from this agent.
        Raises an exception if any argument is missing."""

        if not self.is_ready():
            raise AgentNotReady(
                "Agent needs to be prepared before usage. "
                "You need to set an interpreter, a policy "
                "ensemble as well as a tracker store."
            )

    def create_processor(
        self, preprocessor: Optional[Callable[[Text], Text]] = None
    ) -> MessageProcessor:
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
            message_preprocessor=preprocessor,
        )

    @staticmethod
    def _create_domain(domain: Union[None, Domain, Text]) -> Domain:

        if isinstance(domain, str):
            return Domain.load(domain)
        elif isinstance(domain, Domain):
            return domain
        elif domain is not None:
            raise ValueError(
                "Invalid param `domain`. Expected a path to a domain "
                "specification or a domain instance. But got "
                "type '{}' with value '{}'".format(type(domain), domain)
            )

    @staticmethod
    def create_tracker_store(
        store: Optional["TrackerStore"], domain: Domain
    ) -> "TrackerStore":
        if store is not None:
            store.domain = domain
            return store
        else:
            return InMemoryTrackerStore(domain)

    @staticmethod
    def _create_ensemble(
        policies: Union[List[Policy], PolicyEnsemble, None]
    ) -> Optional[PolicyEnsemble]:
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
                "policies, or a policy ensemble".format(passed_type)
            )

    def _is_form_policy_present(self) -> bool:
        """Check whether form policy is present and used."""

        has_form_policy = self.policy_ensemble and any(
            isinstance(p, FormPolicy) for p in self.policy_ensemble.policies
        )
        return not self.domain or not self.domain.form_names or has_form_policy
