from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Union

import structlog
from jinja2 import Environment, Template, select_autoescape

import rasa.dialogue_understanding.generator.utils
import rasa.shared.utils.io
from rasa.core.config.configuration import Configuration
from rasa.dialogue_understanding.commands import (
    Command,
    SetSlotCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.constants import KEY_MINIMIZE_NUM_CALLS
from rasa.dialogue_understanding.generator import CommandGenerator
from rasa.dialogue_understanding.generator._jinja_filters import to_json_escaped_string
from rasa.dialogue_understanding.generator.constants import (
    DEFAULT_LLM_CONFIG,
    FLOW_RETRIEVAL_ACTIVE_KEY,
    FLOW_RETRIEVAL_FLOW_THRESHOLD,
    FLOW_RETRIEVAL_KEY,
    LLM_CONFIG_KEY,
    TO_JSON_ESCAPED_STRING_JINJA_FILTER,
)
from rasa.dialogue_understanding.generator.flow_retrieval import FlowRetrieval
from rasa.dialogue_understanding.stack.utils import top_flow_frame
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.constants import SetSlotExtractor
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import AgentStarted
from rasa.shared.core.flows import Flow, FlowsList, FlowStep
from rasa.shared.core.flows.steps.collect import CollectInformationFlowStep
from rasa.shared.core.slot_mappings import SlotFillingManager
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import FileIOException, ProviderClientAPIException
from rasa.shared.nlu.constants import FLOWS_IN_PROMPT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.providers.llm.llm_response import LLMResponse, measure_llm_latency
from rasa.shared.utils.constants import (
    LANGFUSE_METADATA_AGENT_ID,
    LANGFUSE_METADATA_COMPONENT_NAME,
    LANGFUSE_METADATA_CUSTOM_METADATA,
    LANGFUSE_METADATA_MODEL_ID,
    LANGFUSE_METADATA_SESSION_ID,
    LANGFUSE_METADATA_TAGS,
)
from rasa.shared.utils.health_check.llm_health_check_mixin import LLMHealthCheckMixin
from rasa.shared.utils.llm import (
    LLMInput,
    allowed_values_for_slot,
    llm_factory,
    resolve_model_client_config,
)
from rasa.utils.log_utils import log_llm

structlogger = structlog.get_logger()


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COMMAND_GENERATOR,
    ],
    is_trainable=True,
)
class LLMBasedCommandGenerator(
    LLMHealthCheckMixin, GraphComponent, CommandGenerator, ABC
):
    """This class provides common functionality for all LLM-based command generators.

    An abstract class defining interface and common functionality
    of an LLM-based command generators.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        self.config = {**self.get_default_config(), **config}
        self.config[LLM_CONFIG_KEY] = resolve_model_client_config(
            self.config.get(LLM_CONFIG_KEY), LLMBasedCommandGenerator.__name__
        )
        self._model_storage = model_storage
        self._resource = resource
        self.flow_retrieval: Optional[FlowRetrieval]

        if self.enabled_flow_retrieval:
            self.flow_retrieval = FlowRetrieval(
                self.config[FLOW_RETRIEVAL_KEY], model_storage, resource
            )
            structlogger.info("llm_based_command_generator.flow_retrieval.enabled")
            self.config[FLOW_RETRIEVAL_KEY] = self.flow_retrieval.config
        else:
            self.flow_retrieval = None

    ### Abstract methods
    @staticmethod
    @abstractmethod
    def get_default_config() -> Dict[str, Any]:
        """The component's default config."""
        pass

    @classmethod
    @abstractmethod
    def load(
        cls: Any,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> "LLMBasedCommandGenerator":
        """Loads trained component (see parent class for full docstring)."""
        pass

    @abstractmethod
    def persist(self) -> None:
        """Persist the component to disk for future loading."""
        pass

    @abstractmethod
    async def predict_commands(
        self,
        message: Message,
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
        **kwargs: Any,
    ) -> List[Command]:
        """Predict commands using the LLM.

        Args:
            message: The message from the user.
            flows: The flows available to the user.
            tracker: The tracker containing the current state of the conversation.
            **kwargs: Keyword arguments for forward compatibility.

        Returns:
            The commands generated by the llm.
        """
        pass

    @abstractmethod
    def parse_commands(
        cls, actions: Optional[str], tracker: DialogueStateTracker, flows: FlowsList
    ) -> List[Command]:
        """Parse the actions returned by the llm into intent and entities.

        Args:
            actions: The actions returned by the llm.
            tracker: The tracker containing the current state of the conversation.
            flows: the list of flows

        Returns:
            The parsed commands.
        """
        pass

    @classmethod
    @abstractmethod
    def fingerprint_addon(cls: Any, config: Dict[str, Any]) -> Optional[str]:
        """Add a fingerprint of the knowledge base for the graph."""
        pass

    ### Shared implementations of GraphComponent parent
    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "LLMBasedCommandGenerator":
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    def train(
        self, training_data: TrainingData, flows: FlowsList, domain: Domain
    ) -> Resource:
        """Trains the LLM-based command generator and prepares flow retrieval data.

        Stores all flows into a vector store.
        """
        self.perform_llm_health_check(
            self.config.get(LLM_CONFIG_KEY),
            self.get_default_llm_config(),
            "llm_based_command_generator.train",
            LLMBasedCommandGenerator.__name__,
        )

        if (
            self.flow_retrieval is None
            and len(flows.user_flows) > FLOW_RETRIEVAL_FLOW_THRESHOLD
        ):
            structlogger.warn(
                "llm_based_command_generator.flow_retrieval.disabled",
                event_info=(
                    f"You have {len(flows.user_flows)} user flows but flow "
                    f"retrieval is disabled. "
                    f"It is recommended to enable flow retrieval if the "
                    f"total number of user flows exceed "
                    f"{FLOW_RETRIEVAL_FLOW_THRESHOLD}. "
                    f"Keeping it disabled can result in deterioration of "
                    f"command generator's functional "
                    f"performance and higher costs because of increased "
                    f"number of tokens in the prompt. For more"
                    "information see:\n"
                    "https://rasa.com/docs/rasa-pro/concepts/dialogue-understanding#how-the-llmcommandgenerator-works"
                ),
            )

        # flow retrieval is populated with only user-defined flows
        try:
            if self.flow_retrieval is not None and not flows.is_empty():
                self.flow_retrieval.populate(flows.user_flows, domain)
        except Exception as e:
            structlogger.error(
                "llm_based_command_generator.train.failed",
                event_info="Flow retrieval store is inaccessible.",
                error=e,
            )
            raise

        self.persist()
        return self._resource

    ### Helper methods
    @property
    def enabled_flow_retrieval(self) -> bool:
        return self.config[FLOW_RETRIEVAL_KEY].get(FLOW_RETRIEVAL_ACTIVE_KEY, True)

    @lru_cache
    def compile_template(self, template: str) -> Template:
        """Compile the prompt template and register custom filters.

        Compiling the template is an expensive operation,
        so we cache the result.
        """
        # Create an environment
        # Autoescaping disabled explicitly for LLM prompt templates rendered from
        # strings (safe, not HTML)
        env = Environment(
            autoescape=select_autoescape(
                disabled_extensions=["jinja2"], default_for_string=False, default=True
            )
        )
        # Register filters
        env.filters[TO_JSON_ESCAPED_STRING_JINJA_FILTER] = to_json_escaped_string

        # Return the template which can leverage registered filters
        return env.from_string(template)

    @classmethod
    def load_prompt_template_from_model_storage(
        cls,
        model_storage: ModelStorage,
        resource: Resource,
        prompt_template_file_name: str,
    ) -> Optional[Text]:
        try:
            with model_storage.read_from(resource) as path:
                return rasa.shared.utils.io.read_file(path / prompt_template_file_name)
        except (FileNotFoundError, FileIOException) as e:
            structlogger.warning(
                "llm_based_command_generator.load_prompt_template.failed",
                error=e,
                resource=resource.name,
            )
        return None

    @classmethod
    def load_flow_retrival(
        cls,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
    ) -> Optional[FlowRetrieval]:
        """Load the FlowRetrieval component if it is enabled in the configuration."""
        enable_flow_retrieval = config.get(FLOW_RETRIEVAL_KEY, {}).get(
            FLOW_RETRIEVAL_ACTIVE_KEY, True
        )
        if enable_flow_retrieval:
            return FlowRetrieval.load(
                config=config.get(FLOW_RETRIEVAL_KEY),
                model_storage=model_storage,
                resource=resource,
            )
        return None

    async def filter_flows(
        self,
        message: Message,
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
    ) -> FlowsList:
        """Filters the available flows.

        Args:
            message: The message from the user.
            flows: The flows available to the user.
            tracker: The tracker containing the current state of the conversation.

        Returns:
            Filtered list of flows.
        """
        # If the flow retrieval is disabled, use the all the provided flows.
        filtered_flows = (
            await self.flow_retrieval.filter_flows(tracker, message, flows)
            if self.flow_retrieval is not None
            else flows
        )
        # Filter flows based on current context (tracker and message)
        # to identify which flows LLM can potentially start.
        if tracker:
            filtered_flows = tracker.get_startable_flows(filtered_flows)
        else:
            filtered_flows = filtered_flows

        # add the filtered flows to the message for evaluation purposes
        message.set(
            FLOWS_IN_PROMPT, list(filtered_flows.user_flow_ids), add_to_output=True
        )
        log_llm(
            logger=structlogger,
            log_module="LLMBasedCommandGenerator",
            log_event="llm_based_command_generator.predict_commands.filtered_flows",
            message=message.data,
            enabled_flow_retrieval=self.flow_retrieval is not None,
            relevant_flows=list(filtered_flows.user_flow_ids),
        )
        return filtered_flows

    @measure_llm_latency
    async def invoke_llm(self, llm_input: LLMInput) -> Optional[LLMResponse]:
        """Use LLM to generate a response.

        Args:
            prompt: The prompt can be,
                - a list of preformatted messages. Each message should be a dictionary
                    with the following keys:
                    - content: The message content.
                    - role: The role of the message (e.g. user or system).
                - a list of messages. Each message is a string and will be formatted
                    as a user message.
                - a single message as a string which will be formatted as user message.

        Returns:
            An LLMResponse object.

        Raises:
            ProviderClientAPIException: If an error occurs during the LLM API call.
        """
        llm = llm_factory(
            self.config.get(LLM_CONFIG_KEY), self.get_default_llm_config()
        )
        try:
            return await llm.acompletion(llm_input.prompt, metadata=llm_input.metadata)
        except Exception as e:
            # unfortunately, langchain does not wrap LLM exceptions which means
            # we have to catch all exceptions here
            structlogger.error("llm_based_command_generator.llm.error", error=e)
            raise ProviderClientAPIException(
                message="LLM call exception", original_exception=e
            )

    def prepare_flows_for_template(
        self,
        flows: FlowsList,
        tracker: DialogueStateTracker,
        add_agent_info: bool = False,
    ) -> List[Dict[str, Any]]:
        """Format data on available flows for insertion into the prompt template.

        Args:
            flows: The flows available to the user.
            tracker: The tracker containing the current state of the conversation.
            add_agent_info: Whether to add agent info to flows or not.

        Returns:
            The inputs for the prompt template.
        """
        result: List[Dict[str, Any]] = []
        for flow in flows.user_flows:
            slots_with_info: List[Dict[str, Any]] = [
                {
                    "name": q.collect,
                    "description": q.description,
                    "allowed_values": allowed_values_for_slot(tracker.slots[q.collect]),
                }
                for q in flow.get_collect_steps()
                if self.is_extractable(q, tracker)
            ]

            agent_info: List[Dict[str, Any]] = []
            if add_agent_info:
                # add information about agents that have been started for this flow
                agent_events = [
                    event
                    for event in tracker.events
                    if isinstance(event, AgentStarted) and event.flow_id == flow.id
                ]
                available_agents = [
                    Configuration.get_instance().available_agents.get_agent_config(
                        event.agent_id
                    )
                    for event in agent_events
                ]
                if available_agents:
                    agent_info = [
                        {
                            "name": available_agent.agent.name,
                            "description": available_agent.agent.description,
                        }
                        for available_agent in available_agents
                        if available_agent is not None
                    ]

            if agent_info:
                result.append(
                    {
                        "name": flow.id,
                        "description": flow.description,
                        "slots": slots_with_info,
                        "agent_info": agent_info,
                    }
                )
            else:
                result.append(
                    {
                        "name": flow.id,
                        "description": flow.description,
                        "slots": slots_with_info,
                    }
                )
        return result

    @staticmethod
    def is_extractable(
        collect_step: CollectInformationFlowStep,
        tracker: DialogueStateTracker,
        current_step: Optional[FlowStep] = None,
    ) -> bool:
        """Check if the `collect` can be filled.

        A collect slot can only be filled if the slot exist
        and either the collect has been asked already or the
        slot has been filled already.

        Args:
            collect_step: The collect_information step.
            tracker: The tracker containing the current state of the conversation.
            current_step: The current step in the flow.

        Returns:
            `True` if the slot can be filled, `False` otherwise.
        """
        slot = tracker.slots.get(collect_step.collect)
        if slot is None:
            return False

        return (
            # we can fill because this is a slot that can be filled ahead of time
            not collect_step.ask_before_filling
            # we can fill because the slot has been filled already
            or slot.has_been_set
            # we can fill because the is currently getting asked
            or (
                current_step is not None
                and isinstance(current_step, CollectInformationFlowStep)
                and current_step.collect == collect_step.collect
            )
        )

    @staticmethod
    def get_slot_value(tracker: DialogueStateTracker, slot_name: str) -> str:
        """Get the slot value from the tracker.

        Args:
            tracker: The tracker containing the current state of the conversation.
            slot_name: The name of the slot.

        Returns:
            The slot value as a string.
        """
        slot_value = tracker.get_slot(slot_name)
        if slot_value is None:
            return "undefined"
        else:
            return str(slot_value)

    def prepare_current_flow_slots_for_template(
        self, top_flow: Flow, current_step: FlowStep, tracker: DialogueStateTracker
    ) -> List[Dict[str, Any]]:
        """Prepare the current flow slots for the template.

        Args:
            top_flow: The top flow.
            current_step: The current step in the flow.
            tracker: The tracker containing the current state of the conversation.

        Returns:
            The slots with values, types, allowed values and a description.
        """
        if top_flow is not None:
            flow_slots = [
                {
                    "name": collect_step.collect,
                    "value": self.get_slot_value(tracker, collect_step.collect),
                    "type": tracker.slots[collect_step.collect].type_name,
                    "allowed_values": allowed_values_for_slot(
                        tracker.slots[collect_step.collect]
                    ),
                    "description": collect_step.description,
                }
                for collect_step in top_flow.get_collect_steps()
                if self.is_extractable(collect_step, tracker, current_step)
            ]
        else:
            flow_slots = []
        return flow_slots

    def prepare_current_slot_for_template(
        self, current_step: FlowStep
    ) -> Tuple[Union[str, None], Union[str, None]]:
        """Prepare the current slot for the template."""
        return (
            (current_step.collect, current_step.description)
            if isinstance(current_step, CollectInformationFlowStep)
            else (None, None)
        )

    @staticmethod
    def _prior_commands_contain_start_flow(prior_commands: List[Command]) -> bool:
        return any(isinstance(command, StartFlowCommand) for command in prior_commands)

    @staticmethod
    def _prior_commands_contain_set_slot_for_active_collect_step(
        prior_commands: List[Command],
        flows: FlowsList,
        tracker: DialogueStateTracker,
    ) -> bool:
        latest_user_frame = top_flow_frame(tracker.stack, ignore_call_frames=False)

        if latest_user_frame is None:
            return False

        active_flow = latest_user_frame.flow(flows)
        active_step = active_flow.step_by_id(latest_user_frame.step_id)

        if not isinstance(active_step, CollectInformationFlowStep):
            return False

        return any(
            command.name == active_step.collect
            for command in prior_commands
            if isinstance(command, SetSlotCommand)
        )

    def _should_skip_llm_call(
        self,
        prior_commands: List[Command],
        flows: FlowsList,
        tracker: DialogueStateTracker,
    ) -> bool:
        """Skip invoking the LLM.

        This returns True if the bot builder sets the property
        KEY_MINIMIZE_NUM_CALLS to True and the prior commands
        either contain a StartFlowCommand or a SetSlot command
        for the current collect step.
        """
        return self.config.get(KEY_MINIMIZE_NUM_CALLS, True) and (
            self._prior_commands_contain_start_flow(prior_commands)
            or self._prior_commands_contain_set_slot_for_active_collect_step(
                prior_commands, flows, tracker
            )
        )

    @staticmethod
    def _check_commands_against_slot_mappings(
        commands: List[Command],
        tracker: DialogueStateTracker,
        domain: Optional[Domain] = None,
    ) -> List[Command]:
        """Check if the LLM-issued slot commands are fillable.

        The LLM-issued slot commands are fillable if the slot
        mappings are satisfied (in particular the mapping conditions).
        """
        if not domain:
            return commands

        llm_fillable_slots = [
            tracker.slots.get(command.name)
            for command in commands
            if isinstance(command, SetSlotCommand)
            and command.extractor == SetSlotExtractor.LLM.value
            and tracker.slots.get(command.name) is not None
        ]

        if not llm_fillable_slots:
            return commands

        slot_filling_manager = SlotFillingManager(domain, tracker)
        slots_to_be_removed = []

        structlogger.debug(
            "command_processor.check_commands_against_slot_mappings.active_flow",
            active_flow=tracker.active_flow,
        )

        for slot in llm_fillable_slots:
            should_fill_slot = False
            for mapping in slot.mappings:  # type: ignore[union-attr]
                should_fill_slot = slot_filling_manager.should_fill_slot(
                    slot.name,  # type: ignore[union-attr]
                    mapping,
                )

                if should_fill_slot:
                    break

            if not should_fill_slot:
                structlogger.debug(
                    "command_processor.check_commands_against_slot_mappings.slot_not_fillable",
                    slot_name=slot.name,  # type: ignore[union-attr]
                )
                slots_to_be_removed.append(slot.name)  # type: ignore[union-attr]

        if not slots_to_be_removed:
            return commands

        filtered_commands = [
            command
            for command in commands
            if not (
                isinstance(command, SetSlotCommand)
                and command.name in slots_to_be_removed
            )
        ]

        return filtered_commands

    def _should_merge_llm_commands(
        self,
        prior_commands: List[Command],
        prior_start_flow_names: Set[str],
    ) -> bool:
        """Check if the LLM current commands should be merged with the prior commands.

        This can be done if there are no prior start flow commands.
        """
        return not prior_start_flow_names

    def _check_start_flow_command_overlap(
        self,
        prior_commands: List[Command],
        commands: List[Command],
        prior_start_flow_names: Set[str],
        current_start_flow_names: Set[str],
    ) -> List[Command]:
        """Prioritize the prior commands over the LLM-issued commands."""
        different_flow_names = current_start_flow_names.difference(
            prior_start_flow_names
        )

        if not different_flow_names or self._should_merge_llm_commands(
            prior_commands, prior_start_flow_names
        ):
            return prior_commands + commands

        # discard the flow names that are different to prior start flow commands
        filtered_commands = [
            command
            for command in commands
            if not isinstance(command, StartFlowCommand)
            or command.flow not in different_flow_names
        ]
        return prior_commands + filtered_commands

    def _filter_slot_commands(
        self,
        prior_commands: List[Command],
        commands: List[Command],
        overlapping_slot_names: Set[str],
    ) -> Tuple[List[Command], List[Command]]:
        """Prioritize prior commands over LLM ones in the case of same slot."""
        filtered_commands = (
            rasa.dialogue_understanding.generator.utils.filter_slot_commands(
                commands, overlapping_slot_names
            )
        )
        return prior_commands, filtered_commands

    @staticmethod
    def get_default_llm_config() -> Dict[str, Any]:
        """Get the default LLM config for the command generator."""
        return DEFAULT_LLM_CONFIG

    def get_llm_tracing_metadata(self, tracker: DialogueStateTracker) -> Dict[str, Any]:
        return {
            LANGFUSE_METADATA_SESSION_ID: tracker.sender_id,
            LANGFUSE_METADATA_TAGS: [self.__class__.__name__],
            LANGFUSE_METADATA_CUSTOM_METADATA: {
                LANGFUSE_METADATA_AGENT_ID: tracker.assistant_id,
                LANGFUSE_METADATA_MODEL_ID: tracker.model_id,
                LANGFUSE_METADATA_COMPONENT_NAME: self.__class__.__name__,
            },
        }
