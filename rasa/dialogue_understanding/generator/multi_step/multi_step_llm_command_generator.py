import importlib.resources
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import structlog
from deprecated import deprecated  # type: ignore[import-untyped]
from jinja2 import Template

import rasa.shared.utils.io
from rasa.dialogue_understanding.commands import (
    CannotHandleCommand,
    ChangeFlowCommand,
    Command,
    ErrorCommand,
    SetSlotCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.commands.can_not_handle_command import (
    DATA_KEY_CANNOT_HANDLE_REASON,
)
from rasa.dialogue_understanding.generator.command_parser import (
    parse_commands as parse_commands_using_command_parsers,
)
from rasa.dialogue_understanding.generator.constants import (
    DEFAULT_LLM_CONFIG,
    FLOW_RETRIEVAL_KEY,
    LLM_CONFIG_KEY,
    USER_INPUT_CONFIG_KEY,
)
from rasa.dialogue_understanding.generator.flow_retrieval import FlowRetrieval
from rasa.dialogue_understanding.generator.llm_based_command_generator import (
    LLMBasedCommandGenerator,
)
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.dialogue_understanding.stack.utils import top_flow_frame, top_user_flow_frame
from rasa.dialogue_understanding.utils import (
    add_commands_to_message_parse_data,
    add_prompt_to_message_parse_data,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import (
    EMBEDDINGS_CONFIG_KEY,
    RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED,
    ROUTE_TO_CALM_SLOT,
)
from rasa.shared.core.flows import Flow, FlowsList, FlowStep
from rasa.shared.core.flows.steps.collect import CollectInformationFlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.utils.constants import (
    LOG_COMPONENT_SOURCE_METHOD_FINGERPRINT_ADDON,
    LOG_COMPONENT_SOURCE_METHOD_INIT,
)
from rasa.shared.utils.io import deep_container_fingerprint, raise_deprecation_warning
from rasa.shared.utils.llm import (
    LLMInput,
    allowed_values_for_slot,
    get_prompt_template,
    resolve_model_client_config,
    sanitize_message_for_prompt,
    tracker_as_readable_transcript,
)

# multistep template keys
HANDLE_FLOWS_KEY = "handle_flows"
FILL_SLOTS_KEY = "fill_slots"

# multistep template file names
HANDLE_FLOWS_PROMPT_FILE_NAME = "handle_flows_prompt.jinja2"
FILL_SLOTS_PROMPT_FILE_NAME = "fill_slots_prompt.jinja2"

# multistep templates
DEFAULT_HANDLE_FLOWS_TEMPLATE = importlib.resources.read_text(
    "rasa.dialogue_understanding.generator.multi_step", "handle_flows_prompt.jinja2"
).strip()
DEFAULT_FILL_SLOTS_TEMPLATE = importlib.resources.read_text(
    "rasa.dialogue_understanding.generator.multi_step", "fill_slots_prompt.jinja2"
).strip()
MULTI_STEP_LLM_COMMAND_GENERATOR_CONFIG_FILE = "config.json"

# dictionary of template names and associated file names and default values
PROMPT_TEMPLATES = {
    HANDLE_FLOWS_KEY: (
        HANDLE_FLOWS_PROMPT_FILE_NAME,
        DEFAULT_HANDLE_FLOWS_TEMPLATE,
    ),
    FILL_SLOTS_KEY: (
        FILL_SLOTS_PROMPT_FILE_NAME,
        DEFAULT_FILL_SLOTS_TEMPLATE,
    ),
}

FILE_PATH_KEY = "file_path"

structlogger = structlog.get_logger()


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COMMAND_GENERATOR,
    ],
    is_trainable=True,
)
@deprecated(
    reason=(
        "The MultiStepLLMCommandGenerator is deprecated and will be removed in "
        "Rasa `4.0.0`."
    )
)
class MultiStepLLMCommandGenerator(LLMBasedCommandGenerator):
    """An multi step command generator using LLM."""

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        prompt_templates: Optional[Dict[Text, Optional[Text]]] = None,
        **kwargs: Any,
    ) -> None:
        raise_deprecation_warning(
            message=(
                "Support for `MultiStepLLMCommandGenerator` will be removed in Rasa "
                "`4.0.0`. Please modify your assistant's configuration to use the "
                "`CompactLLMCommandGenerator` or `SearchReadyLLMCommandGenerator` "
                "instead."
            )
        )

        super().__init__(
            config, model_storage, resource, prompt_templates=prompt_templates, **kwargs
        )

        self._prompts: Dict[Text, Optional[Text]] = {
            HANDLE_FLOWS_KEY: None,
            FILL_SLOTS_KEY: None,
        }
        self._init_prompt_templates(prompt_templates)
        self.trace_prompt_tokens = self.config.get("trace_prompt_tokens", False)

    ### Implementations of LLMBasedCommandGenerator parent
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            "prompt_templates": {},
            USER_INPUT_CONFIG_KEY: None,
            LLM_CONFIG_KEY: None,
            FLOW_RETRIEVAL_KEY: FlowRetrieval.get_default_config(),
        }

    @classmethod
    def load(
        cls: Any,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> "MultiStepLLMCommandGenerator":
        """Loads trained component (see parent class for full docstring)."""
        # Perform health check of the LLM client config
        llm_config = resolve_model_client_config(config.get(LLM_CONFIG_KEY, {}))
        cls.perform_llm_health_check(
            llm_config,
            DEFAULT_LLM_CONFIG,
            "multi_step_llm_command_generator.load",
            MultiStepLLMCommandGenerator.__name__,
        )

        prompts = cls._load_prompt_templates(model_storage, resource)

        # init base command generator
        command_generator = cls(config, model_storage, resource, prompts)
        # load flow retrieval if enabled
        if command_generator.enabled_flow_retrieval:
            command_generator.flow_retrieval = cls.load_flow_retrival(
                command_generator.config, model_storage, resource
            )

        return command_generator

    def persist(self) -> None:
        """Persist this component to disk for future loading."""
        self._persist_prompt_templates()
        self._persist_config()
        if self.flow_retrieval is not None:
            self.flow_retrieval.persist()

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
        prior_commands = self._get_prior_commands(message)

        if tracker is None or flows.is_empty():
            # cannot do anything if there are no flows or no tracker
            return prior_commands

        if self._should_skip_llm_call(prior_commands, flows, tracker):
            return prior_commands

        try:
            commands = await self._predict_commands_with_multi_step(
                message, flows, tracker
            )
            commands = self._clean_up_commands(commands)
            add_commands_to_message_parse_data(
                message, MultiStepLLMCommandGenerator.__name__, commands
            )
        except ProviderClientAPIException:
            # if any step resulted in API exception, the command prediction cannot
            # be completed, "predict" the ErrorCommand
            commands = [ErrorCommand()]

        if not commands and not prior_commands:
            # if for any reason the final list of commands is empty,
            # "predict" CannotHandle
            commands = [CannotHandleCommand()]

        if tracker.has_coexistence_routing_slot:
            # if coexistence feature is used, set the routing slot
            commands += [SetSlotCommand(ROUTE_TO_CALM_SLOT, True)]

        structlogger.debug(
            "multi_step_llm_command_generator.predict_commands.finished",
            commands=commands,
        )

        domain = kwargs.get("domain")
        commands = self._check_commands_against_slot_mappings(commands, tracker, domain)

        return self._check_commands_overlap(prior_commands, commands)

    @classmethod
    def parse_commands(
        cls,
        actions: Optional[str],
        tracker: DialogueStateTracker,
        flows: FlowsList,
        is_handle_flows_prompt: bool = False,
    ) -> List[Command]:
        """Parse the actions returned by the llm into intent and entities.

        Args:
            actions: The actions returned by the llm.
            tracker: The tracker containing the current state of the conversation.
            flows: The list of flows.
            is_handle_flows_prompt: bool

        Returns:
            The parsed commands.
        """
        commands = parse_commands_using_command_parsers(
            actions,
            flows,
            is_handle_flows_prompt=is_handle_flows_prompt,
            additional_commands=[CannotHandleCommand, ChangeFlowCommand],
            data={
                DATA_KEY_CANNOT_HANDLE_REASON: RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED
            },
        )
        if not commands:
            structlogger.debug(
                "multi_step_llm_command_generator.parse_commands",
                message="No commands were parsed from the LLM actions.",
                actions=actions,
            )

        return commands

    ### Helper methods
    @property
    def handle_flows_prompt(self) -> Optional[Text]:
        return self._prompts[HANDLE_FLOWS_KEY]

    @property
    def fill_slots_prompt(self) -> Optional[Text]:
        return self._prompts[FILL_SLOTS_KEY]

    def _init_prompt_templates(self, prompt_templates: Dict[Text, Any]) -> None:
        for key in self._prompts.keys():
            _, default_template = PROMPT_TEMPLATES[key]
            self._prompts[key] = self._resolve_prompt_template(
                prompt_templates, self.config, key, default_template
            )

    @staticmethod
    def _resolve_prompt_template(
        prompt_templates: Optional[Dict[Text, Optional[Text]]],
        config: Dict[Text, Any],
        key: Text,
        default_value: Text,
    ) -> Text:
        """Determines and retrieves a prompt template for a specific step in the
        multistep command generator process using a given key. If the prompt
        associated with the key is missing in both the `prompt_templates` and the
        `config`, this method defaults to using a predefined prompt template. Each key
        is uniquely associated with a distinct step of the command generation process.

        Args:
            prompt_templates: A dictionary of override templates.
            config: The components config that may contain the file paths to the prompt
            templates.
            key: The key for the desired template.
            default_value: The default template to use if no other is found.

        Returns:
            Prompt template.
        """
        if (
            prompt_templates is not None
            and key in prompt_templates
            and prompt_templates[key] is not None
        ):
            return prompt_templates[key]  # type: ignore[return-value]
        return get_prompt_template(
            config.get("prompt_templates", {}).get(key, {}).get(FILE_PATH_KEY),
            default_value,
            log_source_component=MultiStepLLMCommandGenerator.__name__,
            log_source_method=LOG_COMPONENT_SOURCE_METHOD_INIT,
        )

    @classmethod
    def _load_prompt_templates(
        cls, model_storage: ModelStorage, resource: Resource
    ) -> Dict[Text, Text]:
        """Loads persisted prompt templates from the model storage. If a prompt template
        cannot be loaded, default value is used.
        """
        prompts = {}
        for key, (file_name, default_value) in PROMPT_TEMPLATES.items():
            prompt_template = cls.load_prompt_template_from_model_storage(
                model_storage, resource, file_name
            )
            prompts[key] = prompt_template if prompt_template else default_value
        return prompts

    def _persist_prompt_templates(self) -> None:
        """Persist the prompt templates to disk for future loading."""
        with self._model_storage.write_to(self._resource) as path:
            for key, template in self._prompts.items():
                file_name, _ = PROMPT_TEMPLATES[key]
                file_path = path / file_name
                rasa.shared.utils.io.write_text_file(template, file_path)

    def _persist_config(self) -> None:
        """Persist config as a source of truth for resolved clients."""
        with self._model_storage.write_to(self._resource) as path:
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                path / MULTI_STEP_LLM_COMMAND_GENERATOR_CONFIG_FILE, self.config
            )

    async def _predict_commands_with_multi_step(
        self,
        message: Message,
        flows: FlowsList,
        tracker: DialogueStateTracker,
    ) -> List[Command]:
        """Predict commands using the LLM.

        Args:
            message: The message from the user.
            flows: The flows available to the user.
            tracker: The tracker containing the current state of the conversation.

        Returns:
            The commands generated by the llm.

        Raises:
            ProviderClientAPIException: If API calls raised an error.
        """
        # retrieve relevant flows
        filtered_flows = await self.filter_flows(message, flows, tracker)

        # 1st step: Handle active flow
        if tracker.has_active_user_flow:
            commands_from_active_flow = await self._predict_commands_for_active_flow(
                message,
                tracker,
                available_flows=filtered_flows,
                all_flows=flows,
            )
        else:
            commands_from_active_flow = []

        # 2nd step: Check if we need to switch to another flow
        contains_change_flow_command = any(
            isinstance(command, ChangeFlowCommand)
            for command in commands_from_active_flow
        )
        should_change_flows = (
            not commands_from_active_flow or contains_change_flow_command
        )

        if should_change_flows:
            commands_for_handling_flows = (
                await self._predict_commands_for_handling_flows(
                    message,
                    tracker,
                    available_flows=filtered_flows,
                    all_flows=flows,
                )
            )
        else:
            commands_for_handling_flows = []

        if contains_change_flow_command:
            commands_from_active_flow.pop(
                commands_from_active_flow.index(ChangeFlowCommand())
            )

        # 3rd step: Fill slots for started flows
        newly_started_flows = FlowsList(
            [
                flow
                for command in commands_for_handling_flows
                if (
                    isinstance(command, StartFlowCommand)
                    and (flow := filtered_flows.flow_by_id(command.flow)) is not None
                )
            ]
        )

        commands_for_newly_started_flows = (
            await self._predict_commands_for_newly_started_flows(
                message,
                tracker,
                newly_started_flows=newly_started_flows,
                all_flows=flows,
            )
        )

        # concatenate predicted commands
        commands = list(
            set(
                commands_from_active_flow
                + commands_for_handling_flows
                + commands_for_newly_started_flows
            )
        )

        return commands

    async def _predict_commands_for_active_flow(
        self,
        message: Message,
        tracker: DialogueStateTracker,
        available_flows: FlowsList,
        all_flows: FlowsList,
    ) -> List[Command]:
        """Predicts set slots commands for currently active flow.

        Args:
            message: The message from the user.
            tracker: The tracker containing the current state of the conversation.
            available_flows: Startable and active flows.
            all_flows: All flows.

        inputs = self._prepare_inputs(message, tracker, startable_flows, all_flows)

        Returns:
            Predicted commands for the active flow.
        """
        inputs = self._prepare_inputs(message, tracker, available_flows, all_flows)

        if inputs["current_flow"] is None:
            return []

        prompt = Template(self.fill_slots_prompt).render(**inputs).strip()
        structlogger.debug(
            "multi_step_llm_command_generator"
            ".predict_commands_for_active_flow"
            ".prompt_rendered",
            prompt=prompt,
        )
        llm_input = LLMInput(
            prompt=prompt, metadata=self.get_llm_tracing_metadata(tracker)
        )
        response = await self.invoke_llm(llm_input)
        llm_response = LLMResponse.ensure_llm_response(response)
        actions = None
        if llm_response and llm_response.choices:
            actions = llm_response.choices[0]

        structlogger.debug(
            "multi_step_llm_command_generator"
            ".predict_commands_for_active_flow"
            ".actions_generated",
            action_list=actions,
        )

        commands = self.parse_commands(actions, tracker, available_flows)

        if commands:
            add_prompt_to_message_parse_data(
                message=message,
                component_name=MultiStepLLMCommandGenerator.__name__,
                prompt_name="fill_slots_for_active_flow_prompt",
                user_prompt=prompt,
                llm_response=llm_response,
            )

        return commands

    async def _predict_commands_for_handling_flows(
        self,
        message: Message,
        tracker: DialogueStateTracker,
        available_flows: FlowsList,
        all_flows: FlowsList,
    ) -> List[Command]:
        """Predicts commands for starting and canceling flows.

        Args:
            message: The message from the user.
            tracker: The tracker containing the current state of the conversation.
            available_flows: Startable and active flows.
            all_flows: All flows.

        inputs = self._prepare_inputs(message, tracker, startable_flows, all_flows, 2)

        Returns:
            Predicted commands for the starting/canceling flows.
        """
        inputs = self._prepare_inputs(message, tracker, available_flows, all_flows, 2)
        prompt = Template(self.handle_flows_prompt).render(**inputs).strip()
        structlogger.debug(
            "multi_step_llm_command_generator"
            ".predict_commands_for_handling_flows"
            ".prompt_rendered",
            prompt=prompt,
        )

        llm_input = LLMInput(
            prompt=prompt, metadata=self.get_llm_tracing_metadata(tracker)
        )
        response = await self.invoke_llm(llm_input)
        llm_response = LLMResponse.ensure_llm_response(response)
        actions = None
        if llm_response and llm_response.choices:
            actions = llm_response.choices[0]

        structlogger.debug(
            "multi_step_llm_command_generator"
            ".predict_commands_for_handling_flows"
            ".actions_generated",
            action_list=actions,
        )

        commands = self.parse_commands(actions, tracker, available_flows, True)
        # filter out flows that are already started and active
        commands = self._filter_redundant_start_flow_commands(tracker, commands)

        if commands:
            add_prompt_to_message_parse_data(
                message=message,
                component_name=MultiStepLLMCommandGenerator.__name__,
                prompt_name="handle_flows_prompt",
                user_prompt=prompt,
                llm_response=llm_response,
            )

        return commands

    @staticmethod
    def _filter_redundant_start_flow_commands(
        tracker: DialogueStateTracker, commands: List[Command]
    ) -> List[Command]:
        """Filters out StartFlowCommand commands for flows that are already active,
        based on the current tracker state.
        """
        frames = tracker.stack.frames
        active_user_flows = {
            frame.flow_id for frame in frames if isinstance(frame, UserFlowStackFrame)
        }
        commands = [
            command
            for command in commands
            if not (
                isinstance(command, StartFlowCommand)
                and command.flow in active_user_flows
            )
        ]
        return commands

    async def _predict_commands_for_newly_started_flows(
        self,
        message: Message,
        tracker: DialogueStateTracker,
        newly_started_flows: FlowsList,
        all_flows: FlowsList,
    ) -> List[Command]:
        """Predict set slot commands for newly started flows."""
        commands_for_newly_started_flows = []
        for newly_started_flow in newly_started_flows:
            commands_for_newly_started_flows += (
                await self._predict_commands_for_newly_started_flow(
                    newly_started_flow, message, tracker, newly_started_flows
                )
            )
        return commands_for_newly_started_flows

    async def _predict_commands_for_newly_started_flow(
        self,
        newly_started_flow: Flow,
        message: Message,
        tracker: DialogueStateTracker,
        newly_started_flows: FlowsList,
    ) -> List[Command]:
        inputs = self._prepare_inputs_for_single_flow(
            message, tracker, newly_started_flow, max_turns=20
        )

        if len(inputs["flow_slots"]) == 0:
            # return empty if the newly started flow does not have any slots
            return []

        prompt = Template(self.fill_slots_prompt).render(**inputs)
        structlogger.debug(
            "multi_step_llm_command_generator"
            ".predict_commands_for_newly_started_flow"
            ".prompt_rendered",
            flow=newly_started_flow.id,
            prompt=prompt,
        )

        llm_input = LLMInput(
            prompt=prompt, metadata=self.get_llm_tracing_metadata(tracker)
        )
        response = await self.invoke_llm(llm_input)
        llm_response = LLMResponse.ensure_llm_response(response)
        actions = None
        if llm_response and llm_response.choices:
            actions = llm_response.choices[0]

        structlogger.debug(
            "multi_step_llm_command_generator"
            ".predict_commands_for_newly_started_flow"
            ".actions_generated",
            flow=newly_started_flow.id,
            action_list=actions,
        )

        commands = self.parse_commands(actions, tracker, newly_started_flows)

        # filter out all commands that unset values for newly started flow
        commands = [
            command
            for command in commands
            if isinstance(command, SetSlotCommand) and command.value
        ]
        structlogger.debug(
            "multi_step_llm_command_generator"
            ".predict_commands_for_newly_started_flow"
            ".filtered_commands",
            flow=newly_started_flow.id,
            commands=commands,
        )

        if commands:
            add_prompt_to_message_parse_data(
                message=message,
                component_name=MultiStepLLMCommandGenerator.__name__,
                prompt_name="fill_slots_for_new_flow_prompt",
                user_prompt=prompt,
                llm_response=llm_response,
            )

        return commands

    def _prepare_inputs(
        self,
        message: Message,
        tracker: DialogueStateTracker,
        available_flows: FlowsList,
        all_flows: FlowsList,
        max_turns: int = 1,
    ) -> Dict[str, Any]:
        """Prepare input information to be used by prompt template.

        Args:
            message: The message from the user.
            tracker: The tracker containing the current state of the conversation.
            available_flows: Startable and active flows.
            all_flows: All flows.
            max_turns: Max turns of the conversation history between the user
                and the assistant

        Returns:
            Dictionary of inputs.
        """
        top_relevant_frame = top_flow_frame(tracker.stack)
        top_flow = top_relevant_frame.flow(all_flows) if top_relevant_frame else None
        current_step = (
            top_relevant_frame.step(all_flows) if top_relevant_frame else None
        )
        if top_flow is not None:
            flow_slots = self.prepare_current_flow_slots_for_template(
                top_flow, current_step, tracker
            )
            top_flow_is_pattern = top_flow.is_rasa_default_flow
        else:
            flow_slots = []
            top_flow_is_pattern = False

        if top_flow_is_pattern:
            top_user_frame = top_user_flow_frame(tracker.stack)
            top_user_flow = (
                top_user_frame.flow(available_flows) if top_user_frame else None
            )
            top_user_flow_step = (
                top_user_frame.step(available_flows) if top_user_frame else None
            )
            top_user_flow_slots = self.prepare_current_flow_slots_for_template(
                top_user_flow, top_user_flow_step, tracker
            )
        else:
            top_user_flow = None
            top_user_flow_slots = []

        current_slot, current_slot_description = self.prepare_current_slot_for_template(
            current_step
        )
        current_slot_type = None
        current_slot_allowed_values = None
        if current_slot:
            current_slot_type = (
                slot.type_name
                if (slot := tracker.slots.get(current_slot)) is not None
                else None
            )
            current_slot_allowed_values = allowed_values_for_slot(
                tracker.slots.get(current_slot)
            )
        (
            current_conversation,
            latest_user_message,
        ) = self.prepare_conversation_context_for_template(message, tracker, max_turns)

        inputs = {
            "available_flows": self.prepare_flows_for_template(
                available_flows, tracker
            ),
            "current_conversation": current_conversation,
            "current_flow": top_flow.id if top_flow is not None else None,
            "current_slot": current_slot,
            "current_slot_description": current_slot_description,
            "current_slot_type": current_slot_type,
            "current_slot_allowed_values": current_slot_allowed_values,
            "last_user_message": latest_user_message,
            "flow_slots": flow_slots,
            "top_flow_is_pattern": top_flow_is_pattern,
            "top_user_flow": top_user_flow.id if top_user_flow is not None else None,
            "top_user_flow_slots": top_user_flow_slots,
            "flow_active": True,
        }
        return inputs

    def _prepare_inputs_for_single_flow(
        self,
        message: Message,
        tracker: DialogueStateTracker,
        flow: Flow,
        max_turns: int = 1,
    ) -> Dict[Text, Any]:
        flow_slots = self.prepare_current_flow_slots_for_template(
            flow, flow.first_step_in_flow(), tracker
        )
        (
            current_conversation,
            latest_user_message,
        ) = self.prepare_conversation_context_for_template(message, tracker, max_turns)
        inputs = {
            "current_conversation": current_conversation,
            "flow_slots": flow_slots,
            "current_flow": flow.id,
            "last_user_message": latest_user_message,
            "flow_active": False,
        }
        return inputs

    @classmethod
    def fingerprint_addon(cls, config: Dict[str, Any]) -> Optional[str]:
        """Add a fingerprint for the graph."""
        get_prompt_template_log_params = {
            "log_source_component": MultiStepLLMCommandGenerator.__name__,
            "log_source_method": LOG_COMPONENT_SOURCE_METHOD_FINGERPRINT_ADDON,
        }

        handle_flows_template = get_prompt_template(
            config.get("prompt_templates", {})
            .get(HANDLE_FLOWS_KEY, {})
            .get(FILE_PATH_KEY),
            DEFAULT_HANDLE_FLOWS_TEMPLATE,
            **get_prompt_template_log_params,
        )
        fill_slots_template = get_prompt_template(
            config.get("prompt_templates", {})
            .get(FILL_SLOTS_KEY, {})
            .get(FILE_PATH_KEY),
            DEFAULT_FILL_SLOTS_TEMPLATE,
            **get_prompt_template_log_params,
        )

        llm_config = resolve_model_client_config(
            config.get(LLM_CONFIG_KEY), MultiStepLLMCommandGenerator.__name__
        )
        embedding_config = resolve_model_client_config(
            config.get(FLOW_RETRIEVAL_KEY, {}).get(EMBEDDINGS_CONFIG_KEY),
            FlowRetrieval.__name__,
        )

        return deep_container_fingerprint(
            [handle_flows_template, fill_slots_template, llm_config, embedding_config]
        )

    @staticmethod
    def prepare_conversation_context_for_template(
        message: Message, tracker: DialogueStateTracker, max_turns: int = 20
    ) -> Tuple[Text, Text]:
        current_conversation = tracker_as_readable_transcript(
            tracker, max_turns=max_turns
        )
        latest_user_message = sanitize_message_for_prompt(message.get(TEXT))
        current_conversation += f"\nUSER: {latest_user_message}"
        return current_conversation, latest_user_message

    def prepare_current_flow_slots_for_template(
        self, top_flow: Flow, current_step: FlowStep, tracker: DialogueStateTracker
    ) -> List[Dict[Text, Any]]:
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

    @staticmethod
    def prepare_current_slot_for_template(
        current_step: FlowStep,
    ) -> Tuple[Union[str, None], Union[str, None]]:
        """Prepare the current slot for the template."""
        return (
            (current_step.collect, current_step.description)
            if isinstance(current_step, CollectInformationFlowStep)
            else (None, None)
        )

    @staticmethod
    def _clean_up_commands(commands: List[Command]) -> List[Command]:
        """Cleans the list of commands by removing CannotHandleCommand,
        if it exists and there are other commands in the list.
        """
        other_commands_count = sum(
            not isinstance(command, CannotHandleCommand) for command in commands
        )

        if other_commands_count == len(commands):
            # no cannot handle command found
            return commands

        if other_commands_count:
            # remove cannot handle commands
            return [
                command
                for command in commands
                if not isinstance(command, CannotHandleCommand)
            ]

        # only cannot handle commands present
        return [CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED)]
