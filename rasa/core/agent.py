import logging
import os
import shutil
import tempfile
import uuid
from asyncio import CancelledError
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union

import aiohttp
from sanic import Sanic

import rasa
import rasa.utils.io
import rasa.core.utils
from rasa.constants import (
    DEFAULT_DOMAIN_PATH,
    LEGACY_DOCS_BASE_URL,
    ENV_SANIC_BACKLOG,
    DEFAULT_CORE_SUBDIRECTORY_NAME,
)
from rasa.core import constants, jobs, training
from rasa.core.channels.channel import InputChannel, OutputChannel, UserMessage
from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT
from rasa.core.domain import Domain
from rasa.core.exceptions import AgentNotReady
from rasa.core.interpreter import NaturalLanguageInterpreter, RegexInterpreter
from rasa.core.lock_store import LockStore, InMemoryLockStore
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.core.policies.ensemble import PolicyEnsemble, SimplePolicyEnsemble
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.policy import Policy
from rasa.core.processor import MessageProcessor
from rasa.core.tracker_store import (
    InMemoryTrackerStore,
    TrackerStore,
    FailSafeTrackerStore,
)
from rasa.core.trackers import DialogueStateTracker
from rasa.exceptions import ModelNotFound
from rasa.importers.importer import TrainingDataImporter
from rasa.model import (
    get_model_subdirectories,
    get_latest_model,
    unpack_model,
    get_model,
)
from rasa.nlu.utils import is_url
from rasa.utils.common import raise_warning, update_sanic_log_level
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


async def load_from_server(agent: "Agent", model_server: EndpointConfig) -> "Agent":
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


def _load_and_set_updated_model(
    agent: "Agent", model_directory: Text, fingerprint: Text
):
    """Load the persisted model into memory and set the model on the agent."""

    logger.debug(f"Found new model with fingerprint {fingerprint}. Loading...")

    core_path, nlu_path = get_model_subdirectories(model_directory)

    if nlu_path:
        from rasa.core.interpreter import RasaNLUInterpreter

        interpreter = RasaNLUInterpreter(model_directory=nlu_path)
    else:
        interpreter = (
            agent.interpreter if agent.interpreter is not None else RegexInterpreter()
        )

    domain = None
    if core_path:
        domain_path = os.path.join(os.path.abspath(core_path), DEFAULT_DOMAIN_PATH)
        domain = Domain.load(domain_path)

    try:
        policy_ensemble = None
        if core_path:
            policy_ensemble = PolicyEnsemble.load(core_path)
        agent.update_model(
            domain, policy_ensemble, fingerprint, interpreter, model_directory
        )
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

    model_directory_and_fingerprint = await _pull_model_and_fingerprint(
        model_server, agent.fingerprint
    )
    if model_directory_and_fingerprint:
        model_directory, new_model_fingerprint = model_directory_and_fingerprint
        _load_and_set_updated_model(agent, model_directory, new_model_fingerprint)
    else:
        logger.debug(f"No new model found at URL {model_server.url}")


async def _pull_model_and_fingerprint(
    model_server: EndpointConfig, fingerprint: Optional[Text]
) -> Optional[Tuple[Text, Text]]:
    """Queries the model server.

    Returns the temporary model directory and value of the response's <ETag> header
    which contains the model hash. Returns `None` if no new model is found.
    """

    headers = {"If-None-Match": fingerprint}

    logger.debug(f"Requesting model from server {model_server.url}...")

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
                    return None
                elif resp.status == 404:
                    logger.debug(
                        "Model server could not find a model at the requested "
                        "endpoint '{}'. It's possible that no model has been "
                        "trained, or that the requested tag hasn't been "
                        "assigned.".format(model_server.url)
                    )
                    return None
                elif resp.status != 200:
                    logger.debug(
                        "Tried to fetch model from server, but server response "
                        "status code is {}. We'll retry later..."
                        "".format(resp.status)
                    )
                    return None

                model_directory = tempfile.mkdtemp()
                rasa.utils.io.unarchive(await resp.read(), model_directory)
                logger.debug(
                    "Unzipped model to '{}'".format(os.path.abspath(model_directory))
                )

                # get the new fingerprint
                new_fingerprint = resp.headers.get("ETag")
                # return new tmp model directory and new fingerprint
                return model_directory, new_fingerprint

        except aiohttp.ClientError as e:
            logger.debug(
                "Tried to fetch model from server, but "
                "couldn't reach server. We'll retry later... "
                "Error: {}.".format(e)
            )
            return None


async def _run_model_pulling_worker(
    model_server: EndpointConfig, agent: "Agent"
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
        args=[model_server, agent],
        id="pull-model-from-server",
        replace_existing=True,
    )


async def load_agent(
    model_path: Optional[Text] = None,
    model_server: Optional[EndpointConfig] = None,
    remote_storage: Optional[Text] = None,
    interpreter: Optional[NaturalLanguageInterpreter] = None,
    generator: Union[EndpointConfig, NaturalLanguageGenerator] = None,
    tracker_store: Optional[TrackerStore] = None,
    lock_store: Optional[LockStore] = None,
    action_endpoint: Optional[EndpointConfig] = None,
):
    try:
        if model_server is not None:
            return await load_from_server(
                Agent(
                    interpreter=interpreter,
                    generator=generator,
                    tracker_store=tracker_store,
                    lock_store=lock_store,
                    action_endpoint=action_endpoint,
                    model_server=model_server,
                    remote_storage=remote_storage,
                ),
                model_server,
            )

        elif remote_storage is not None:
            return Agent.load_from_remote_storage(
                remote_storage,
                model_path,
                interpreter=interpreter,
                generator=generator,
                tracker_store=tracker_store,
                lock_store=lock_store,
                action_endpoint=action_endpoint,
                model_server=model_server,
            )

        elif model_path is not None and os.path.exists(model_path):
            return Agent.load_local_model(
                model_path,
                interpreter=interpreter,
                generator=generator,
                tracker_store=tracker_store,
                lock_store=lock_store,
                action_endpoint=action_endpoint,
                model_server=model_server,
                remote_storage=remote_storage,
            )

        else:
            raise_warning("No valid configuration given to load agent.")
            return None

    except Exception as e:
        logger.error(f"Could not load model due to {e}.")
        raise


class Agent:
    """The Agent class provides a convenient interface for the most important
     Rasa functionality.

     This includes training, handling messages, loading a dialogue model,
     getting the next action, and handling a channel."""

    def __init__(
        self,
        domain: Union[Text, Domain, None] = None,
        policies: Union[PolicyEnsemble, List[Policy], None] = None,
        interpreter: Optional[NaturalLanguageInterpreter] = None,
        generator: Union[EndpointConfig, NaturalLanguageGenerator, None] = None,
        tracker_store: Optional[TrackerStore] = None,
        lock_store: Optional[LockStore] = None,
        action_endpoint: Optional[EndpointConfig] = None,
        fingerprint: Optional[Text] = None,
        model_directory: Optional[Text] = None,
        model_server: Optional[EndpointConfig] = None,
        remote_storage: Optional[Text] = None,
        path_to_model_archive: Optional[Text] = None,
    ):
        # Initializing variables with the passed parameters.
        self.domain = self._create_domain(domain)
        self.policy_ensemble = self._create_ensemble(policies)

        if self.domain is not None:
            self.domain.add_requested_slot()
            self.domain.add_knowledge_base_slots()
            self.domain.add_categorical_slot_default_value()

        PolicyEnsemble.check_domain_ensemble_compatibility(
            self.policy_ensemble, self.domain
        )

        self.interpreter = NaturalLanguageInterpreter.create(interpreter)

        self.nlg = NaturalLanguageGenerator.create(generator, self.domain)
        self.tracker_store = self.create_tracker_store(tracker_store, self.domain)
        self.lock_store = self._create_lock_store(lock_store)
        self.action_endpoint = action_endpoint

        self._set_fingerprint(fingerprint)
        self.model_directory = model_directory
        self.model_server = model_server
        self.remote_storage = remote_storage
        self.path_to_model_archive = path_to_model_archive

    def update_model(
        self,
        domain: Optional[Domain],
        policy_ensemble: Optional[PolicyEnsemble],
        fingerprint: Optional[Text],
        interpreter: Optional[NaturalLanguageInterpreter] = None,
        model_directory: Optional[Text] = None,
    ) -> None:
        self.domain = self._create_domain(domain)
        self.policy_ensemble = policy_ensemble

        if interpreter:
            self.interpreter = NaturalLanguageInterpreter.create(interpreter)

        self._set_fingerprint(fingerprint)

        # update domain on all instances
        self.tracker_store.domain = domain
        if hasattr(self.nlg, "templates"):
            self.nlg.templates = domain.templates if domain else {}

        self.model_directory = model_directory

    @classmethod
    def load(
        cls,
        model_path: Text,
        interpreter: Optional[NaturalLanguageInterpreter] = None,
        generator: Union[EndpointConfig, NaturalLanguageGenerator] = None,
        tracker_store: Optional[TrackerStore] = None,
        lock_store: Optional[LockStore] = None,
        action_endpoint: Optional[EndpointConfig] = None,
        model_server: Optional[EndpointConfig] = None,
        remote_storage: Optional[Text] = None,
        path_to_model_archive: Optional[Text] = None,
    ) -> "Agent":
        """Load a persisted model from the passed path."""
        try:
            if not model_path:
                raise ModelNotFound("No path specified.")
            elif not os.path.exists(model_path):
                raise ModelNotFound(f"No file or directory at '{model_path}'.")
            elif os.path.isfile(model_path):
                model_path = get_model(model_path)
        except ModelNotFound:
            raise ValueError(
                "You are trying to load a MODEL from '{}', which is not possible. \n"
                "The model path should be a 'tar.gz' file or a directory "
                "containing the various model files in the sub-directories 'core' "
                "and 'nlu'. \n\nIf you want to load training data instead of "
                "a model, use `agent.load_data(...)` instead.".format(model_path)
            )

        core_model, nlu_model = get_model_subdirectories(model_path)

        if not interpreter and nlu_model:
            interpreter = NaturalLanguageInterpreter.create(nlu_model)

        domain = None
        ensemble = None

        if core_model:
            domain = Domain.load(os.path.join(core_model, DEFAULT_DOMAIN_PATH))
            ensemble = PolicyEnsemble.load(core_model) if core_model else None

            # ensures the domain hasn't changed between test and train
            domain.compare_with_specification(core_model)

        return cls(
            domain=domain,
            policies=ensemble,
            interpreter=interpreter,
            generator=generator,
            tracker_store=tracker_store,
            lock_store=lock_store,
            action_endpoint=action_endpoint,
            model_directory=model_path,
            model_server=model_server,
            remote_storage=remote_storage,
            path_to_model_archive=path_to_model_archive,
        )

    def is_core_ready(self) -> bool:
        """Check if all necessary components and policies are ready to use the agent.
        """
        return self.is_ready() and self.policy_ensemble is not None

    def is_ready(self) -> bool:
        """Check if all necessary components are instantiated to use agent.

        Policies might not be available, if this is an NLU only agent."""

        return self.tracker_store is not None and self.interpreter is not None

    async def parse_message_using_nlu_interpreter(
        self, message_data: Text, tracker: DialogueStateTracker = None
    ) -> Dict[Text, Any]:
        """Handles message text and intent payload input messages.

        The return value of this function is parsed_data.

        Args:
            message_data (Text): Contain the received message in text or\
            intent payload format.
            tracker (DialogueStateTracker): Contains the tracker to be\
            used by the interpreter.

        Returns:
            The parsed message.

            Example:

                {\
                    "text": '/greet{"name":"Rasa"}',\
                    "intent": {"name": "greet", "confidence": 1.0},\
                    "intent_ranking": [{"name": "greet", "confidence": 1.0}],\
                    "entities": [{"entity": "name", "start": 6,\
                                  "end": 21, "value": "Rasa"}],\
                }

        """

        processor = self.create_processor()
        message = UserMessage(message_data)
        return await processor._parse_message(message, tracker)

    async def handle_message(
        self,
        message: UserMessage,
        message_preprocessor: Optional[Callable[[Text], Text]] = None,
        **kwargs,
    ) -> Optional[List[Dict[Text, Any]]]:
        """Handle a single message."""

        if not isinstance(message, UserMessage):
            raise_warning(
                "Passing a text to `agent.handle_message(...)` is "
                "deprecated. Rather use `agent.handle_text(...)`.",
                DeprecationWarning,
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

        async with self.lock_store.lock(message.sender_id):
            return await processor.handle_message(message)

    # noinspection PyUnusedLocal
    async def predict_next(
        self, sender_id: Text, **kwargs: Any
    ) -> Optional[Dict[Text, Any]]:
        """Handle a single message."""

        processor = self.create_processor()
        return await processor.predict_next(sender_id)

    # noinspection PyUnusedLocal
    async def log_message(
        self,
        message: UserMessage,
        message_preprocessor: Optional[Callable[[Text], Text]] = None,
        **kwargs: Any,
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
        return await processor.execute_action(
            sender_id, action, output_channel, self.nlg, policy, confidence
        )

    async def trigger_intent(
        self,
        intent_name: Text,
        entities: List[Dict[Text, Any]],
        output_channel: OutputChannel,
        tracker: DialogueStateTracker,
    ) -> None:
        """Trigger a user intent, e.g. triggered by an external event."""

        processor = self.create_processor()
        await processor.trigger_external_user_uttered(
            intent_name, entities, tracker, output_channel,
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
            >>> agent = Agent.load("examples/restaurantbot/models/current")
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

    def _max_history(self) -> int:
        """Find maximum max_history."""

        max_histories = [
            policy.featurizer.max_history
            for policy in self.policy_ensemble.policies
            if hasattr(policy.featurizer, "max_history")
        ]

        return max(max_histories or [0])

    def _are_all_featurizers_using_a_max_history(self) -> bool:
        """Check if all featurizers are MaxHistoryTrackerFeaturizer."""

        def has_max_history_featurizer(policy):
            return policy.featurizer and hasattr(policy.featurizer, "max_history")

        for p in self.policy_ensemble.policies:
            if p.featurizer and not has_max_history_featurizer(p):
                return False
        return True

    async def load_data(
        self,
        training_resource: Union[Text, TrainingDataImporter],
        remove_duplicates: bool = True,
        unique_last_num_states: Optional[int] = None,
        augmentation_factor: int = 50,
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
            raise_warning(
                f"unique_last_num_states={unique_last_num_states} but "
                f"maximum max_history={max_history}. "
                f"Possibility of data loss. "
                f"It is recommended to set "
                f"unique_last_num_states to "
                f"at least maximum max_history."
            )

        return await training.load_data(
            training_resource,
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
        if not self.is_core_ready():
            raise AgentNotReady("Can't train without a policy ensemble.")

        # deprecation tests
        if kwargs.get("featurizer"):
            raise Exception(
                "Passing `featurizer` "
                "to `agent.train(...)` is not supported anymore. "
                "Pass appropriate featurizer directly "
                "to the policy configuration instead. More info "
                "{}/core/migrations.html".format(LEGACY_DOCS_BASE_URL)
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
                "{}/core/migrations.html".format(LEGACY_DOCS_BASE_URL)
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

        logger.debug(f"Agent trainer got kwargs: {kwargs}")

        self.policy_ensemble.train(training_trackers, self.domain, **kwargs)
        self._set_fingerprint()

    def handle_channels(
        self,
        channels: List[InputChannel],
        http_port: int = constants.DEFAULT_SERVER_PORT,
        route: Text = "/webhooks/",
        cors: Union[Text, List[Text], None] = None,
    ) -> Sanic:
        """Start a webserver attaching the input channels and handling msgs."""

        from rasa.core import run

        raise_warning(
            "Using `handle_channels` is deprecated. "
            "Please use `rasa.run(...)` or see "
            "`rasa.core.run.configure_app(...)` if you want to implement "
            "this on a more detailed level.",
            DeprecationWarning,
        )

        app = run.configure_app(channels, cors, None, enable_api=False, route=route)

        app.agent = self

        update_sanic_log_level()

        app.run(
            host="0.0.0.0",
            port=http_port,
            backlog=int(os.environ.get(ENV_SANIC_BACKLOG, "100")),
            workers=rasa.core.utils.number_of_sanic_workers(self.lock_store),
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

    def persist(self, model_path: Text) -> None:
        """Persists this agent into a directory for later loading and usage."""

        if not self.is_core_ready():
            raise AgentNotReady("Can't persist without a policy ensemble.")

        if not model_path.endswith(DEFAULT_CORE_SUBDIRECTORY_NAME):
            model_path = os.path.join(model_path, DEFAULT_CORE_SUBDIRECTORY_NAME)

        self._clear_model_directory(model_path)

        self.policy_ensemble.persist(model_path)
        self.domain.persist(os.path.join(model_path, DEFAULT_DOMAIN_PATH))
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

    def create_processor(
        self, preprocessor: Optional[Callable[[Text], Text]] = None
    ) -> MessageProcessor:
        """Instantiates a processor based on the set state of the agent."""
        # Checks that the interpreter and tracker store are set and
        # creates a processor
        if not self.is_ready():
            raise AgentNotReady(
                "Agent needs to be prepared before usage. You need to set an "
                "interpreter and a tracker store."
            )

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
    def _create_domain(domain: Union[Domain, Text, None]) -> Domain:

        if isinstance(domain, str):
            domain = Domain.load(domain)
            domain.check_missing_templates()
            return domain
        elif isinstance(domain, Domain):
            return domain
        elif domain is None:
            return Domain.empty()
        else:
            raise ValueError(
                "Invalid param `domain`. Expected a path to a domain "
                "specification or a domain instance. But got "
                "type '{}' with value '{}'".format(type(domain), domain)
            )

    @staticmethod
    def create_tracker_store(
        store: Optional[TrackerStore], domain: Domain
    ) -> TrackerStore:
        if store is not None:
            store.domain = domain
            tracker_store = store
        else:
            tracker_store = InMemoryTrackerStore(domain)

        return FailSafeTrackerStore(tracker_store)

    @staticmethod
    def _create_lock_store(store: Optional[LockStore]) -> LockStore:
        if store is not None:
            return store

        return InMemoryLockStore()

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
                "policies, or a policy ensemble.".format(passed_type)
            )

    @staticmethod
    def load_local_model(
        model_path: Text,
        interpreter: Optional[NaturalLanguageInterpreter] = None,
        generator: Union[EndpointConfig, NaturalLanguageGenerator] = None,
        tracker_store: Optional[TrackerStore] = None,
        lock_store: Optional[LockStore] = None,
        action_endpoint: Optional[EndpointConfig] = None,
        model_server: Optional[EndpointConfig] = None,
        remote_storage: Optional[Text] = None,
    ) -> "Agent":
        if os.path.isfile(model_path):
            model_archive = model_path
        else:
            model_archive = get_latest_model(model_path)

        if model_archive is None:
            raise_warning(f"Could not load local model in '{model_path}'.")
            return Agent()

        working_directory = tempfile.mkdtemp()
        unpacked_model = unpack_model(model_archive, working_directory)

        return Agent.load(
            unpacked_model,
            interpreter=interpreter,
            generator=generator,
            tracker_store=tracker_store,
            lock_store=lock_store,
            action_endpoint=action_endpoint,
            model_server=model_server,
            remote_storage=remote_storage,
            path_to_model_archive=model_archive,
        )

    @staticmethod
    def load_from_remote_storage(
        remote_storage: Text,
        model_name: Text,
        interpreter: Optional[NaturalLanguageInterpreter] = None,
        generator: Union[EndpointConfig, NaturalLanguageGenerator] = None,
        tracker_store: Optional[TrackerStore] = None,
        lock_store: Optional[LockStore] = None,
        action_endpoint: Optional[EndpointConfig] = None,
        model_server: Optional[EndpointConfig] = None,
    ) -> Optional["Agent"]:
        from rasa.nlu.persistor import get_persistor

        persistor = get_persistor(remote_storage)

        if persistor is not None:
            target_path = tempfile.mkdtemp()
            persistor.retrieve(model_name, target_path)

            return Agent.load(
                target_path,
                interpreter=interpreter,
                generator=generator,
                tracker_store=tracker_store,
                lock_store=lock_store,
                action_endpoint=action_endpoint,
                model_server=model_server,
                remote_storage=remote_storage,
            )

        return None
