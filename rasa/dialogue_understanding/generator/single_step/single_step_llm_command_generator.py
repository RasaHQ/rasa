import importlib.resources
from typing import Any, Dict, List, Optional, Text

import structlog

import rasa.shared.utils.io
from rasa.dialogue_understanding.commands import (
    CannotHandleCommand,
    Command,
    ErrorCommand,
    SetSlotCommand,
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
from rasa.dialogue_understanding.stack.utils import top_flow_frame
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
    PROMPT_CONFIG_KEY,
    PROMPT_TEMPLATE_CONFIG_KEY,
    ROUTE_TO_CALM_SLOT,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.nlu.constants import LLM_COMMANDS, LLM_PROMPT, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.utils.io import deep_container_fingerprint
from rasa.shared.utils.llm import (
    get_prompt_template,
    resolve_model_client_config,
    sanitize_message_for_prompt,
    tracker_as_readable_transcript,
)
from rasa.utils.beta import BetaNotEnabledException, ensure_beta_feature_is_enabled
from rasa.utils.log_utils import log_llm

COMMAND_PROMPT_FILE_NAME = "command_prompt.jinja2"

DEFAULT_COMMAND_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.dialogue_understanding.generator.single_step",
    "command_prompt_template.jinja2",
)
SINGLE_STEP_LLM_COMMAND_GENERATOR_CONFIG_FILE = "config.json"

structlogger = structlog.get_logger()


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COMMAND_GENERATOR,
    ],
    is_trainable=True,
)
class SingleStepLLMCommandGenerator(LLMBasedCommandGenerator):
    """A single step LLM-based command generator."""

    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        prompt_template: Optional[Text] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config,
            model_storage,
            resource,
            prompt_template=prompt_template,
            **kwargs,
        )

        # Set the prompt template
        if config.get(PROMPT_CONFIG_KEY):
            structlogger.warning(
                "single_step_llm_command_generator.init",
                event_info=(
                    "The config parameter 'prompt' is deprecated "
                    "and will be removed in Rasa 4.0.0. "
                    "Please use the config parameter 'prompt_template' instead. "
                ),
            )
        config_prompt = (
            config.get(PROMPT_CONFIG_KEY)
            or config.get(PROMPT_TEMPLATE_CONFIG_KEY)
            or None
        )
        self.prompt_template = prompt_template or get_prompt_template(
            config_prompt,
            DEFAULT_COMMAND_PROMPT_TEMPLATE,
        )

        self.trace_prompt_tokens = self.config.get("trace_prompt_tokens", False)
        self.repeat_command_enabled = self.is_repeat_command_enabled()

    ### Implementations of LLMBasedCommandGenerator parent
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            PROMPT_CONFIG_KEY: None,  # Legacy
            PROMPT_TEMPLATE_CONFIG_KEY: None,
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
    ) -> "SingleStepLLMCommandGenerator":
        """Loads trained component (see parent class for full docstring)."""
        # Perform health check of the LLM API endpoint
        llm_config = resolve_model_client_config(config.get(LLM_CONFIG_KEY, {}))
        cls.perform_llm_health_check(
            llm_config,
            DEFAULT_LLM_CONFIG,
            "single_step_llm_command_generator.load",
            SingleStepLLMCommandGenerator.__name__,
        )

        # load prompt template from the model storage.
        prompt_template = cls.load_prompt_template_from_model_storage(
            model_storage, resource, COMMAND_PROMPT_FILE_NAME
        )

        # init base command generator
        command_generator = cls(config, model_storage, resource, prompt_template)
        # load flow retrieval if enabled
        if command_generator.enabled_flow_retrieval:
            command_generator.flow_retrieval = cls.load_flow_retrival(
                command_generator.config, model_storage, resource
            )

        return command_generator

    def persist(self) -> None:
        """Persist this component to disk for future loading."""
        self._persist_prompt_template()
        self._persist_config()
        if self.flow_retrieval is not None:
            self.flow_retrieval.persist()

    def _persist_prompt_template(self) -> None:
        """Persist prompt template for future loading."""
        with self._model_storage.write_to(self._resource) as path:
            rasa.shared.utils.io.write_text_file(
                self.prompt_template, path / COMMAND_PROMPT_FILE_NAME
            )

    def _persist_config(self) -> None:
        """Persist config as a source of truth for resolved clients."""
        with self._model_storage.write_to(self._resource) as path:
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                path / SINGLE_STEP_LLM_COMMAND_GENERATOR_CONFIG_FILE, self.config
            )

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
        if tracker is None or flows.is_empty():
            # cannot do anything if there are no flows or no tracker
            return []

        try:
            commands = await self._predict_commands(message, flows, tracker)
        except ProviderClientAPIException:
            # if command predictions resulted in API exception
            # "predict" the ErrorCommand
            commands = [ErrorCommand()]

        if not commands:
            # no commands are parsed or there's an invalid command
            structlogger.warning(
                "single_step_llm_command_generator.predict_commands",
                message="No commands were predicted as the LLM response could "
                "not be parsed or the LLM responded with an invalid command."
                "Returning a CannotHandleCommand instead.",
            )
            commands = [CannotHandleCommand()]

        if tracker.has_coexistence_routing_slot:
            # if coexistence feature is used, set the routing slot
            commands += [SetSlotCommand(ROUTE_TO_CALM_SLOT, True)]

        log_llm(
            logger=structlogger,
            log_module="SingleStepLLMCommandGenerator",
            log_event="llm_command_generator.predict_commands.finished",
            commands=commands,
        )

        return commands

    async def _predict_commands(
        self,
        message: Message,
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
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
        # retrieve flows
        filtered_flows = await self.filter_flows(message, flows, tracker)

        flow_prompt = self.render_template(message, tracker, filtered_flows, flows)
        log_llm(
            logger=structlogger,
            log_module="SingleStepLLMCommandGenerator",
            log_event="llm_command_generator.predict_commands.prompt_rendered",
            prompt=flow_prompt,
        )

        response = await self.invoke_llm(flow_prompt)
        llm_response = LLMResponse.ensure_llm_response(response)
        # The check for 'None' maintains compatibility with older versions
        # of LLMCommandGenerator. In previous implementations, 'invoke_llm'
        # might return 'None' to indicate a failure to generate actions.
        if llm_response is None or not llm_response.choices:
            return [ErrorCommand()]

        action_list = llm_response.choices[0]

        log_llm(
            logger=structlogger,
            log_module="SingleStepLLMCommandGenerator",
            log_event="llm_command_generator.predict_commands.actions_generated",
            action_list=action_list,
        )

        commands = self.parse_commands(action_list, tracker, flows)

        self._update_message_parse_data_for_fine_tuning(message, commands, flow_prompt)
        add_commands_to_message_parse_data(
            message, SingleStepLLMCommandGenerator.__name__, commands
        )
        add_prompt_to_message_parse_data(
            message=message,
            component_name=SingleStepLLMCommandGenerator.__name__,
            prompt_name="command_generator_prompt",
            user_prompt=flow_prompt,
            llm_response=llm_response,
        )

        return commands

    @staticmethod
    def _update_message_parse_data_for_fine_tuning(
        message: Message, commands: List[Command], prompt: str
    ) -> None:
        from rasa.llm_fine_tuning.annotation_module import preparing_fine_tuning_data

        if preparing_fine_tuning_data:
            # Add commands and prompt to the message object in order to create
            # prompt -> commands pairs for fine-tuning
            message.set(
                LLM_COMMANDS,
                [command.as_dict() for command in commands],
                add_to_output=True,
            )
            message.set(LLM_PROMPT, prompt, add_to_output=True)

    @classmethod
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
        commands = parse_commands_using_command_parsers(actions, flows)
        if not commands:
            structlogger.debug(
                "single_step_llm_command_generator.parse_commands",
                message="No commands were parsed from the LLM actions.",
                actions=actions,
            )

        return commands

    @classmethod
    def fingerprint_addon(cls: Any, config: Dict[str, Any]) -> Optional[str]:
        """Add a fingerprint for the graph."""
        config_prompt = (
            config.get(PROMPT_CONFIG_KEY)
            or config.get(PROMPT_TEMPLATE_CONFIG_KEY)
            or None
        )
        prompt_template = get_prompt_template(
            config_prompt,
            DEFAULT_COMMAND_PROMPT_TEMPLATE,
        )
        llm_config = resolve_model_client_config(
            config.get(LLM_CONFIG_KEY), SingleStepLLMCommandGenerator.__name__
        )
        embedding_config = resolve_model_client_config(
            config.get(FLOW_RETRIEVAL_KEY, {}).get(EMBEDDINGS_CONFIG_KEY),
            FlowRetrieval.__name__,
        )
        return deep_container_fingerprint(
            [prompt_template, llm_config, embedding_config]
        )

    ### Helper methods
    def render_template(
        self,
        message: Message,
        tracker: DialogueStateTracker,
        startable_flows: FlowsList,
        all_flows: FlowsList,
    ) -> str:
        """Render the jinja template to create the prompt for the LLM.

        Args:
            message: The current message from the user.
            tracker: The tracker containing the current state of the conversation.
            startable_flows: The flows startable at this point in time by the user.
            all_flows: all flows present in the assistant

        Returns:
            The rendered prompt template.
        """
        # need to make this distinction here because current step of the
        # top_calling_frame would be the call step, but we need the collect step from
        # the called frame. If no call is active calling and called frame are the same.
        top_calling_frame = top_flow_frame(tracker.stack)
        top_called_frame = top_flow_frame(tracker.stack, ignore_call_frames=False)

        top_flow = top_calling_frame.flow(all_flows) if top_calling_frame else None
        current_step = top_called_frame.step(all_flows) if top_called_frame else None

        flow_slots = self.prepare_current_flow_slots_for_template(
            top_flow, current_step, tracker
        )
        current_slot, current_slot_description = self.prepare_current_slot_for_template(
            current_step
        )
        current_conversation = tracker_as_readable_transcript(tracker)
        latest_user_message = sanitize_message_for_prompt(message.get(TEXT))
        current_conversation += f"\nUSER: {latest_user_message}"

        inputs = {
            "available_flows": self.prepare_flows_for_template(
                startable_flows, tracker
            ),
            "current_conversation": current_conversation,
            "flow_slots": flow_slots,
            "current_flow": top_flow.id if top_flow is not None else None,
            "current_slot": current_slot,
            "current_slot_description": current_slot_description,
            "user_message": latest_user_message,
            "is_repeat_command_enabled": self.repeat_command_enabled,
        }

        return self.compile_template(self.prompt_template).render(**inputs)

    def is_repeat_command_enabled(self) -> bool:
        """Check for feature flag"""
        RASA_PRO_BETA_REPEAT_COMMAND_ENV_VAR_NAME = "RASA_PRO_BETA_REPEAT_COMMAND"
        try:
            ensure_beta_feature_is_enabled(
                "Repeat Command",
                env_flag=RASA_PRO_BETA_REPEAT_COMMAND_ENV_VAR_NAME,
            )
        except BetaNotEnabledException:
            return False

        return True
