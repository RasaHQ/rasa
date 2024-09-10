import importlib.resources
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Text

import structlog
from jinja2 import Template

import rasa.shared.utils.io
from rasa.dialogue_understanding.commands import (
    Command,
    ErrorCommand,
    SetSlotCommand,
    CancelFlowCommand,
    StartFlowCommand,
    HumanHandoffCommand,
    ChitChatAnswerCommand,
    SkipQuestionCommand,
    KnowledgeAnswerCommand,
    ClarifyCommand,
    CannotHandleCommand,
)
from rasa.dialogue_understanding.commands.change_flow_command import ChangeFlowCommand
from rasa.dialogue_understanding.generator.constants import (
    LLM_CONFIG_KEY,
    USER_INPUT_CONFIG_KEY,
    FLOW_RETRIEVAL_KEY,
)
from rasa.dialogue_understanding.generator.flow_retrieval import FlowRetrieval
from rasa.dialogue_understanding.generator.llm_based_command_generator import (
    LLMBasedCommandGenerator,
)
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.dialogue_understanding.stack.utils import (
    top_flow_frame,
    top_user_flow_frame,
    user_flows_on_the_stack,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.flows import FlowStep, Flow, FlowsList
from rasa.shared.core.flows.steps.collect import CollectInformationFlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.utils.io import deep_container_fingerprint
from rasa.shared.utils.llm import (
    get_prompt_template,
    tracker_as_readable_transcript,
    sanitize_message_for_prompt,
    allowed_values_for_slot,
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
        # persist prompt template
        self._persist_prompt_templates()
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
            commands = await self._predict_commands_with_multi_step(
                message, flows, tracker
            )
            commands = self._clean_up_commands(commands)
        except ProviderClientAPIException:
            # if any step resulted in API exception, the command prediction cannot
            # be completed, "predict" the ErrorCommand
            commands = [ErrorCommand()]

        if not commands:
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

        return commands

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
        if not actions:
            return []

        commands: List[Command] = []

        slot_set_re = re.compile(
            r"""SetSlot\((\"?[a-zA-Z_][a-zA-Z0-9_-]*?\"?), ?(.*)\)"""
        )
        start_flow_re = re.compile(r"StartFlow\(([a-zA-Z0-9_-]+?)\)")
        change_flow_re = re.compile(r"ChangeFlow\(\)")
        cancel_flow_re = re.compile(r"CancelFlow\(\)")
        chitchat_re = re.compile(r"ChitChat\(\)")
        skip_question_re = re.compile(r"SkipQuestion\(\)")
        knowledge_re = re.compile(r"SearchAndReply\(\)")
        humand_handoff_re = re.compile(r"HumanHandoff\(\)")
        clarify_re = re.compile(r"Clarify\(([\"\'a-zA-Z0-9_, ]+)\)")
        cannot_handle_re = re.compile(r"CannotHandle\(\)")

        for action in actions.strip().splitlines():
            if is_handle_flows_prompt:
                if (
                    len(commands) >= 2
                    or len(commands) == 1
                    and isinstance(commands[0], ClarifyCommand)
                ):
                    break

            if cannot_handle_re.search(action):
                commands.append(
                    CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED)
                )
            if match := slot_set_re.search(action):
                slot_name = cls.clean_extracted_value(match.group(1).strip())
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
                    flow
                    for flow in options
                    if flow in flows.user_flow_ids
                    and flow not in user_flows_on_the_stack(tracker.stack)
                ]
                if len(valid_options) == 1:
                    commands.extend(cls.start_flow_by_name(valid_options[0], flows))
                elif 1 < len(valid_options) <= 5:
                    commands.append(ClarifyCommand(valid_options))
            elif change_flow_re.search(action):
                commands.append(ChangeFlowCommand())

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

        actions = await self.invoke_llm(prompt)
        structlogger.debug(
            "multi_step_llm_command_generator"
            ".predict_commands_for_active_flow"
            ".actions_generated",
            action_list=actions,
        )

        commands = self.parse_commands(actions, tracker, available_flows)
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

        actions = await self.invoke_llm(prompt)
        structlogger.debug(
            "multi_step_llm_command_generator"
            ".predict_commands_for_handling_flows"
            ".actions_generated",
            action_list=actions,
        )

        commands = self.parse_commands(actions, tracker, available_flows, True)
        # filter out flows that are already started and active
        commands = self._filter_redundant_start_flow_commands(tracker, commands)

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

        actions = await self.invoke_llm(prompt)
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
        handle_flows_template = get_prompt_template(
            config.get("prompt_templates", {})
            .get(HANDLE_FLOWS_KEY, {})
            .get(FILE_PATH_KEY),
            DEFAULT_HANDLE_FLOWS_TEMPLATE,
        )
        fill_slots_template = get_prompt_template(
            config.get("prompt_templates", {})
            .get(FILL_SLOTS_KEY, {})
            .get(FILE_PATH_KEY),
            DEFAULT_FILL_SLOTS_TEMPLATE,
        )
        return deep_container_fingerprint(
            [
                handle_flows_template,
                fill_slots_template,
            ]
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
