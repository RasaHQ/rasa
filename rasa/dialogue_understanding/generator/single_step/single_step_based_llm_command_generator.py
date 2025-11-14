import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Text

import structlog

import rasa.shared.utils.io
from rasa.agents.utils import (
    get_active_agent_info,
    get_completed_agents_info,
)
from rasa.core.config.configuration import Configuration
from rasa.dialogue_understanding.commands import (
    CannotHandleCommand,
    Command,
    ErrorCommand,
    SetSlotCommand,
)
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.generator import LLMBasedCommandGenerator
from rasa.dialogue_understanding.generator.command_parser import (
    parse_commands as parse_commands_using_command_parsers,
)
from rasa.dialogue_understanding.generator.command_parser_validator import (
    CommandParserValidatorSingleton,
)
from rasa.dialogue_understanding.generator.constants import (
    COMMAND_PROMPT_FILE_NAME,
    FLOW_RETRIEVAL_KEY,
    LLM_BASED_COMMAND_GENERATOR_CONFIG_FILE,
    LLM_CONFIG_KEY,
    USER_INPUT_CONFIG_KEY,
)
from rasa.dialogue_understanding.generator.flow_retrieval import FlowRetrieval
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
    DEFAULT_INCLUDE_DATE_TIME,
    DEFAULT_TIMEZONE,
    EMBEDDINGS_CONFIG_KEY,
    INCLUDE_DATE_TIME_CONFIG_KEY,
    PROMPT_TEMPLATE_CONFIG_KEY,
    ROUTE_TO_CALM_SLOT,
    TIMEZONE_CONFIG_KEY,
)
from rasa.shared.core.constants import MOCKED_DATETIME_SLOT
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.nlu.constants import LLM_COMMANDS, LLM_PROMPT, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.utils.constants import (
    LOG_COMPONENT_SOURCE_METHOD_FINGERPRINT_ADDON,
    LOG_COMPONENT_SOURCE_METHOD_INIT,
)
from rasa.shared.utils.datetime_utils import (
    resolve_datetime,
    validate_datetime_configuration,
)
from rasa.shared.utils.io import deep_container_fingerprint
from rasa.shared.utils.llm import (
    LLMInput,
    allowed_values_for_slot,
    resolve_model_client_config,
    sanitize_message_for_prompt,
    tracker_as_readable_transcript,
)
from rasa.utils.log_utils import log_llm

structlogger = structlog.get_logger()


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COMMAND_GENERATOR,
    ],
    is_trainable=True,
)
class SingleStepBasedLLMCommandGenerator(LLMBasedCommandGenerator, ABC):
    """Abstract class single step based LLM command generator."""

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

        # Get the prompt template from the config or the default prompt template.
        self.prompt_template = self._resolve_component_prompt_template(
            self.config,
            prompt_template,
            log_context=LOG_COMPONENT_SOURCE_METHOD_INIT,
            log_source_component=self.__class__.__name__,
        )

        # Set the command syntax version.
        CommandSyntaxManager.set_syntax_version(
            self.get_component_command_syntax_version()
        )

        self.trace_prompt_tokens = self.config.get("trace_prompt_tokens", False)

        # Datetime configuration
        self.include_date_time = self.config.get(
            INCLUDE_DATE_TIME_CONFIG_KEY, DEFAULT_INCLUDE_DATE_TIME
        )
        self.timezone = self.config.get(TIMEZONE_CONFIG_KEY, DEFAULT_TIMEZONE)

        # Validate datetime configuration
        validate_datetime_configuration(
            self.include_date_time,
            self.timezone,
            TIMEZONE_CONFIG_KEY in self.config,
            self.__class__.__name__,
        )

    ### Implementations of LLMBasedCommandGenerator parent
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            PROMPT_TEMPLATE_CONFIG_KEY: None,  # TODO: remove in Rasa 4.0.0
            USER_INPUT_CONFIG_KEY: None,
            LLM_CONFIG_KEY: None,
            FLOW_RETRIEVAL_KEY: FlowRetrieval.get_default_config(),
        }

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
                path / LLM_BASED_COMMAND_GENERATOR_CONFIG_FILE, self.config
            )

    @classmethod
    def load(
        cls: Any,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> "SingleStepBasedLLMCommandGenerator":
        """Loads trained component (see parent class for full docstring)."""
        # Perform health check of the LLM API endpoint
        llm_config = resolve_model_client_config(config.get(LLM_CONFIG_KEY, {}))
        cls.perform_llm_health_check(
            llm_config,
            cls.get_default_llm_config(),
            "llm_based_command_generator.load",
            cls.__name__,
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
            commands = await self._predict_commands(message, flows, tracker)
        except ProviderClientAPIException:
            # if command predictions resulted in API exception
            # "predict" the ErrorCommand
            commands = [ErrorCommand()]
            structlogger.warning(
                "llm_command_generator.predict_commands.api_exception",
                event_info=(
                    "ProviderClientAPIException occurred while predicting commands."
                ),
                commands=commands,  # no PII
            )

        if not commands and not prior_commands:
            # no commands are parsed or there's an invalid command
            structlogger.warning(
                "llm_command_generator.predict_commands",
                message="No commands were predicted as the LLM response could "
                "not be parsed or the LLM responded with an invalid command. "
                "Returning a CannotHandleCommand instead.",
            )
            commands = [CannotHandleCommand()]

        if tracker.has_coexistence_routing_slot:
            # if coexistence feature is used, set the routing slot
            commands += [SetSlotCommand(ROUTE_TO_CALM_SLOT, True)]

        log_llm(
            logger=structlogger,
            log_module=self.__class__.__name__,
            log_event="llm_command_generator.predict_commands.finished",
            commands=commands,
            event_info="Commands predictd by the Command Generator",
            highlight=True,
        )

        domain = kwargs.get("domain")
        commands = self._check_commands_against_slot_mappings(commands, tracker, domain)

        return self._check_commands_overlap(prior_commands, commands)

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
            log_module=self.__class__.__name__,
            log_event="llm_command_generator.predict_commands.prompt_rendered",
            prompt=flow_prompt,
        )

        llm_input = LLMInput(
            prompt=flow_prompt, metadata=self.get_llm_tracing_metadata(tracker)
        )
        response = await self.invoke_llm(llm_input)
        llm_response = LLMResponse.ensure_llm_response(response)
        # The check for 'None' maintains compatibility with older versions
        # of LLMCommandGenerator. In previous implementations, 'invoke_llm'
        # might return 'None' to indicate a failure to generate actions.
        if llm_response is None or not llm_response.choices:
            structlogger.warning(
                "llm_command_generator.predict_commands.no_actions_generated",
                event_info=(
                    "No actions were generated by the LLM. Returning an ErrorCommand."
                ),
            )
            return [ErrorCommand()]

        action_list = llm_response.choices[0]

        log_llm(
            logger=structlogger,
            log_module=self.__class__.__name__,
            log_event="llm_command_generator.predict_commands.actions_generated",
            action_list=action_list,
        )

        commands = self.parse_commands(action_list, tracker, flows)

        if CommandParserValidatorSingleton.should_validate_command_parser():
            CommandParserValidatorSingleton.validate_if_commands_are_parsed_from_llm_response(
                commands, action_list
            )

        self._update_message_parse_data_for_fine_tuning(message, commands, flow_prompt)
        add_commands_to_message_parse_data(message, self.__class__.__name__, commands)
        add_prompt_to_message_parse_data(
            message=message,
            component_name=self.__class__.__name__,
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
            structlogger.warning(
                f"{cls.__name__}.parse_commands",
                message="No commands were parsed from the LLM actions.",
                actions=actions,
            )

        return commands

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
        has_agents = Configuration.get_instance().available_agents.has_agents()
        current_conversation = tracker_as_readable_transcript(
            tracker, highlight_agent_turns=has_agents
        )
        latest_user_message = sanitize_message_for_prompt(message.get(TEXT))
        current_conversation += f"\nUSER: {latest_user_message}"

        inputs: Dict[str, Any] = {
            "available_flows": self.prepare_flows_for_template(
                startable_flows,
                tracker,
                add_agent_info=has_agents,
            ),
            "current_conversation": current_conversation,
            "flow_slots": flow_slots,
            "current_flow": top_flow.id if top_flow is not None else None,
            "current_slot": current_slot,
            "current_slot_description": current_slot_description,
            "current_slot_type": current_slot_type,
            "current_slot_allowed_values": current_slot_allowed_values,
            "user_message": latest_user_message,
        }
        if has_agents:
            inputs["active_agent"] = (
                get_active_agent_info(tracker, top_flow.id) if top_flow else None
            )
            inputs["completed_agents"] = get_completed_agents_info(tracker)

        # Add current datetime if enabled
        if self.include_date_time:
            mocked_datetime_value = tracker.get_slot(MOCKED_DATETIME_SLOT)
            inputs["current_datetime"] = resolve_datetime(
                mocked_datetime_value, timezone=self.timezone
            )

        return self.compile_template(self.prompt_template).render(**inputs)

    @classmethod
    def fingerprint_addon(cls: Any, config: Dict[str, Any]) -> Optional[str]:
        """Add a fingerprint for the graph."""
        # Get the default prompt template based on the model name
        llm_config = resolve_model_client_config(
            config.get(LLM_CONFIG_KEY), cls.__name__
        )
        embedding_config = resolve_model_client_config(
            config.get(FLOW_RETRIEVAL_KEY, {}).get(EMBEDDINGS_CONFIG_KEY),
            FlowRetrieval.__name__,
        )

        # Create a copy of the config to avoid modifying the original config
        # and update the llm config with the resolved llm config.
        _config_copy = copy.deepcopy(config)
        _config_copy[LLM_CONFIG_KEY] = llm_config
        prompt_template = cls._resolve_component_prompt_template(
            _config_copy,
            log_context=LOG_COMPONENT_SOURCE_METHOD_FINGERPRINT_ADDON,
            log_source_component=cls.__name__,
        )

        return deep_container_fingerprint(
            [prompt_template, llm_config, embedding_config]
        )

    @staticmethod
    @abstractmethod
    def get_component_command_syntax_version() -> CommandSyntaxVersion:
        """Get the command syntax version for the command generator."""
        pass

    @classmethod
    @abstractmethod
    def _resolve_component_prompt_template(
        cls: Any,
        config: Dict[str, Any],
        prompt_template: Optional[str] = None,
        log_context: Optional[Literal["init", "fingerprint_addon"]] = None,
        log_source_component: Optional[str] = "SingleStepBasedLLMCommandGenerator",
    ) -> Optional[str]:
        """Get the prompt template from the config or the default prompt template."""
        pass
