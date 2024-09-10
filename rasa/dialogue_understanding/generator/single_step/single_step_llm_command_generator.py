import importlib.resources
import re
from typing import Dict, Any, List, Optional, Text

import structlog

import rasa.shared.utils.io
from rasa.dialogue_understanding.commands import (
    Command,
    ErrorCommand,
    SetSlotCommand,
    CancelFlowCommand,
    HumanHandoffCommand,
    ChitChatAnswerCommand,
    SkipQuestionCommand,
    KnowledgeAnswerCommand,
    ClarifyCommand,
    CannotHandleCommand,
)
from rasa.dialogue_understanding.generator.constants import (
    LLM_CONFIG_KEY,
    USER_INPUT_CONFIG_KEY,
    FLOW_RETRIEVAL_KEY,
)
from rasa.dialogue_understanding.generator.flow_retrieval import (
    FlowRetrieval,
)
from rasa.dialogue_understanding.generator.llm_based_command_generator import (
    LLMBasedCommandGenerator,
)
from rasa.dialogue_understanding.stack.utils import top_flow_frame
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import (
    ROUTE_TO_CALM_SLOT,
    PROMPT_CONFIG_KEY,
    PROMPT_TEMPLATE_CONFIG_KEY,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.nlu.constants import TEXT, LLM_COMMANDS, LLM_PROMPT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.utils.io import deep_container_fingerprint
from rasa.shared.utils.llm import (
    get_prompt_template,
    tracker_as_readable_transcript,
    sanitize_message_for_prompt,
)
from rasa.utils.log_utils import log_llm

COMMAND_PROMPT_FILE_NAME = "command_prompt.jinja2"

DEFAULT_COMMAND_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.dialogue_understanding.generator.single_step",
    "command_prompt_template.jinja2",
)

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
        # persist prompt template
        with self._model_storage.write_to(self._resource) as path:
            rasa.shared.utils.io.write_text_file(
                self.prompt_template, path / COMMAND_PROMPT_FILE_NAME
            )
        # persist flow retrieval
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

        action_list = await self.invoke_llm(flow_prompt)
        # The check for 'None' maintains compatibility with older versions
        # of LLMCommandGenerator. In previous implementations, 'invoke_llm'
        # might return 'None' to indicate a failure to generate actions.
        if action_list is None:
            return [ErrorCommand()]

        log_llm(
            logger=structlogger,
            log_module="SingleStepLLMCommandGenerator",
            log_event="llm_command_generator.predict_commands.actions_generated",
            action_list=action_list,
        )

        commands = self.parse_commands(action_list, tracker, flows)

        self._update_message_parse_data_for_fine_tuning(message, commands, flow_prompt)

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
        if actions is None:
            return []

        commands: List[Command] = []

        slot_set_re = re.compile(r"""SetSlot\(([a-zA-Z_][a-zA-Z0-9_-]*?), ?(.*)\)""")
        start_flow_re = re.compile(r"StartFlow\(([a-zA-Z0-9_-]+?)\)")
        cancel_flow_re = re.compile(r"CancelFlow\(\)")
        chitchat_re = re.compile(r"ChitChat\(\)")
        skip_question_re = re.compile(r"SkipQuestion\(\)")
        knowledge_re = re.compile(r"SearchAndReply\(\)")
        humand_handoff_re = re.compile(r"HumanHandoff\(\)")
        clarify_re = re.compile(r"Clarify\(([a-zA-Z0-9_, ]+)\)")

        for action in actions.strip().splitlines():
            if match := slot_set_re.search(action):
                slot_name = match.group(1).strip()
                slot_value = cls.clean_extracted_value(match.group(2))
                # error case where the llm tries to start a flow using a slot set
                if slot_name == "flow_name":
                    commands.extend(cls.start_flow_by_name(slot_value, flows))
                else:
                    typed_slot_value = cls.get_nullable_slot_value(slot_value)
                    commands.append(
                        SetSlotCommand(name=slot_name, value=typed_slot_value)
                    )
            elif match := start_flow_re.search(action):
                flow_name = match.group(1).strip()
                commands.extend(cls.start_flow_by_name(flow_name, flows))
            elif cancel_flow_re.search(action):
                commands.append(CancelFlowCommand())
            elif chitchat_re.search(action):
                commands.append(ChitChatAnswerCommand())
            elif skip_question_re.search(action):
                commands.append(SkipQuestionCommand())
            elif knowledge_re.search(action):
                commands.append(KnowledgeAnswerCommand())
            elif humand_handoff_re.search(action):
                commands.append(HumanHandoffCommand())
            elif match := clarify_re.search(action):
                options = sorted([opt.strip() for opt in match.group(1).split(",")])
                valid_options = [
                    flow for flow in options if flow in flows.user_flow_ids
                ]
                if len(set(valid_options)) == 1:
                    commands.extend(cls.start_flow_by_name(valid_options[0], flows))
                elif len(valid_options) > 1:
                    commands.append(ClarifyCommand(valid_options))

        return commands

    @classmethod
    def fingerprint_addon(cls: Any, config: Dict[str, Any]) -> Optional[str]:
        """Add a fingerprint of the knowledge base for the graph."""
        config_prompt = (
            config.get(PROMPT_CONFIG_KEY)
            or config.get(PROMPT_TEMPLATE_CONFIG_KEY)
            or None
        )
        prompt_template = get_prompt_template(
            config_prompt,
            DEFAULT_COMMAND_PROMPT_TEMPLATE,
        )
        return deep_container_fingerprint(prompt_template)

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
        }

        return self.compile_template(self.prompt_template).render(**inputs)
