import importlib.resources
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Text
from functools import lru_cache
import rasa.shared.utils.io
import structlog
from jinja2 import Template
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
from rasa.dialogue_understanding.generator import CommandGenerator
from rasa.dialogue_understanding.generator.flow_retrieval import FlowRetrieval
from rasa.dialogue_understanding.stack.utils import top_flow_frame
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import FlowStep, Flow, FlowsList
from rasa.shared.core.flows.steps.collect import CollectInformationFlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import FileIOException
from rasa.shared.nlu.constants import FLOWS_IN_PROMPT
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils.io import deep_container_fingerprint
from rasa.shared.utils.llm import (
    DEFAULT_OPENAI_CHAT_MODEL_NAME_ADVANCED,
    DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    get_prompt_template,
    llm_factory,
    tracker_as_readable_transcript,
    sanitize_message_for_prompt,
    allowed_values_for_slot,
)
from rasa.utils.log_utils import log_llm

COMMAND_PROMPT_FILE_NAME = "command_prompt.jinja2"

DEFAULT_COMMAND_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.dialogue_understanding.generator", "command_prompt_template.jinja2"
)

DEFAULT_LLM_CONFIG = {
    "_type": "openai",
    "request_timeout": 7,
    "temperature": 0.0,
    "model_name": DEFAULT_OPENAI_CHAT_MODEL_NAME_ADVANCED,
    "max_tokens": DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
}

LLM_CONFIG_KEY = "llm"
USER_INPUT_CONFIG_KEY = "user_input"

FLOW_RETRIEVAL_KEY = "flow_retrieval"
FLOW_RETRIEVAL_ACTIVE_KEY = "active"

structlogger = structlog.get_logger()


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COMMAND_GENERATOR,
    ],
    is_trainable=True,
)
class LLMCommandGenerator(GraphComponent, CommandGenerator):
    """An LLM-based command generator."""

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            "prompt": None,
            USER_INPUT_CONFIG_KEY: None,
            LLM_CONFIG_KEY: None,
            FLOW_RETRIEVAL_KEY: FlowRetrieval.get_default_config(),
        }

    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        prompt_template: Optional[Text] = None,
    ) -> None:
        super().__init__(config)
        self.config = {**self.get_default_config(), **config}

        self.prompt_template = prompt_template or get_prompt_template(
            config.get("prompt"),
            DEFAULT_COMMAND_PROMPT_TEMPLATE,
        )

        self._model_storage = model_storage
        self._resource = resource
        self.trace_prompt_tokens = self.config.get("trace_prompt_tokens", False)

        self.flow_retrieval: Optional[FlowRetrieval]

        if self.enabled_flow_retrieval:
            self.flow_retrieval = FlowRetrieval(
                self.config[FLOW_RETRIEVAL_KEY], model_storage, resource
            )
            structlogger.info("llm_command_generator.flow_retrieval.enabled")
        else:
            self.flow_retrieval = None
            structlogger.warn(
                "llm_command_generator.flow_retrieval.disabled",
                event_info=(
                    "Disabling flow retrieval can cause issues when there are a "
                    "large number of flows to be included in the prompt. For more"
                    "information see:\n"
                    "https://rasa.com/docs/rasa-pro/concepts/dialogue-understanding#how-the-llmcommandgenerator-works"
                ),
            )

    @property
    def enabled_flow_retrieval(self) -> bool:
        return self.config[FLOW_RETRIEVAL_KEY].get(FLOW_RETRIEVAL_ACTIVE_KEY, True)

    @lru_cache
    def _compile_template(self, template: str) -> Template:
        """Compile the prompt template.

        Compiling the template is an expensive operation,
        so we cache the result."""
        return Template(template)

    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "LLMCommandGenerator":
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    @classmethod
    def load(
        cls,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> "LLMCommandGenerator":
        """Loads trained component (see parent class for full docstring)."""
        # load prompt template from the model storage.
        prompt_template = cls._load_prompt_template_from_model_storage(
            model_storage, resource
        )
        # init base command generator
        command_generator = cls(config, model_storage, resource, prompt_template)
        # load flow retrieval if enabled
        if command_generator.enabled_flow_retrieval:
            command_generator.flow_retrieval = cls._load_flow_retrival(
                command_generator.config, model_storage, resource
            )
        return command_generator

    @classmethod
    def _load_prompt_template_from_model_storage(
        cls, model_storage: ModelStorage, resource: Resource
    ) -> Optional[Text]:
        try:
            with model_storage.read_from(resource) as path:
                return rasa.shared.utils.io.read_file(path / COMMAND_PROMPT_FILE_NAME)
        except (FileNotFoundError, FileIOException) as e:
            structlogger.warning(
                "llm_command_generator.load_prompt_template.failed",
                error=e,
                resource=resource.name,
            )
        return None

    @classmethod
    def _load_flow_retrival(
        cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource
    ) -> Optional[FlowRetrieval]:
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

    def train(
        self, training_data: TrainingData, flows: FlowsList, domain: Domain
    ) -> Resource:
        """Train the llm command generator. Stores all flows into a vector store."""
        # flow retrieval is populated with only user-defined flows
        try:
            if self.flow_retrieval is not None:
                self.flow_retrieval.populate(flows.user_flows, domain)
        except Exception as e:
            structlogger.error(
                "llm_command_generator.train.failed",
                event_info=("Flow retrieval store isinaccessible."),
                error=e,
            )
            raise
        self.persist()
        return self._resource

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
    ) -> List[Command]:
        """Predict commands using the LLM.

        Args:
            message: The message from the user.
            flows: The flows available to the user.
            tracker: The tracker containing the current state of the conversation.

        Returns:
            The commands generated by the llm.
        """
        if tracker is None or flows.is_empty():
            # cannot do anything if there are no flows or no tracker
            return []

        commands: List[Command]

        # retrieve flows
        try:
            # If the flow retrieval is disabled, use the all the provided flows.
            filtered_flows = (
                await self.flow_retrieval.filter_flows(tracker, message, flows)
                if self.flow_retrieval is not None
                else flows
            )
            # Filter flows based on current context (tracker and message)
            # to identify which flows LLM can potentially start.
            filtered_flows = tracker.get_startable_flows(filtered_flows)
        except Exception:
            # e.g. in case of API problems (are being logged by the flow retrieval)
            commands = [ErrorCommand()]
            # if coexistence feature is used, set the routing slot
            if tracker.has_coexistence_routing_slot:
                commands += [SetSlotCommand(ROUTE_TO_CALM_SLOT, True)]
            return commands

        # add the filtered flows to the message for evaluation purposes
        message.set(
            FLOWS_IN_PROMPT, list(filtered_flows.user_flow_ids), add_to_output=True
        )
        log_llm(
            logger=structlogger,
            log_module="LLMCommandGenerator",
            log_event="llm_command_generator.predict_commands.filtered_flows",
            message=message.data[TEXT],
            enabled_flow_retrieval=self.flow_retrieval is not None,
            relevant_flows=list(filtered_flows.user_flow_ids),
        )

        flow_prompt = self.render_template(message, tracker, filtered_flows, flows)
        log_llm(
            logger=structlogger,
            log_module="LLMCommandGenerator",
            log_event="llm_command_generator.predict_commands.prompt_rendered",
            prompt=flow_prompt,
        )

        action_list = await self._generate_action_list_using_llm(flow_prompt)
        log_llm(
            logger=structlogger,
            log_module="LLMCommandGenerator",
            log_event="llm_command_generator.predict_commands.actions_generated",
            action_list=action_list,
        )

        if action_list is None:
            # if action_list is None, we couldn't get any response from the LLM
            commands = [ErrorCommand()]
        else:
            commands = self.parse_commands(action_list, tracker, flows)
            if not commands:
                # no commands are parsed or there's an invalid command
                commands = [CannotHandleCommand()]

        # if coexistence feature is used, set the routing slot
        if tracker.has_coexistence_routing_slot:
            commands += [SetSlotCommand(ROUTE_TO_CALM_SLOT, True)]

        log_llm(
            logger=structlogger,
            log_module="LLMCommandGenerator",
            log_event="llm_command_generator.predict_commands.finished",
            commands=commands,
        )
        return commands

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

        return self._compile_template(self.prompt_template).render(**inputs)

    async def _generate_action_list_using_llm(self, prompt: str) -> Optional[str]:
        """Use LLM to generate a response.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            The generated text.
        """
        llm = llm_factory(self.config.get(LLM_CONFIG_KEY), DEFAULT_LLM_CONFIG)

        try:
            return await llm.apredict(prompt)
        except Exception as e:
            # unfortunately, langchain does not wrap LLM exceptions which means
            # we have to catch all exceptions here
            structlogger.error("llm_command_generator.llm.error", error=e)
            return None

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
                if len(valid_options) >= 1:
                    commands.append(ClarifyCommand(valid_options))

        return commands

    @staticmethod
    def start_flow_by_name(flow_name: str, flows: FlowsList) -> List[Command]:
        """Start a flow by name.

        If the flow does not exist, no command is returned.
        """
        if flow_name in flows.user_flow_ids:
            return [StartFlowCommand(flow=flow_name)]
        else:
            structlogger.debug(
                "llm_command_generator.flow.start_invalid_flow_id", flow=flow_name
            )
            return []

    @staticmethod
    def is_none_value(value: str) -> bool:
        """Check if the value is a none value."""
        return value in {
            "[missing information]",
            "[missing]",
            "None",
            "undefined",
            "null",
        }

    @staticmethod
    def clean_extracted_value(value: str) -> str:
        """Clean up the extracted value from the llm."""
        # replace any combination of single quotes, double quotes, and spaces
        # from the beginning and end of the string
        return value.strip("'\" ")

    @classmethod
    def get_nullable_slot_value(cls, slot_value: str) -> Union[str, None]:
        """Get the slot value or None if the value is a none value.

        Args:
            slot_value: the value to coerce

        Returns:
            The slot value or None if the value is a none value.
        """
        return slot_value if not cls.is_none_value(slot_value) else None

    def prepare_flows_for_template(
        self, flows: FlowsList, tracker: DialogueStateTracker
    ) -> List[Dict[str, Any]]:
        """Format data on available flows for insertion into the prompt template.

        Args:
            flows: The flows available to the user.
            tracker: The tracker containing the current state of the conversation.

        Returns:
            The inputs for the prompt template.
        """
        result = []
        for flow in flows.user_flows:
            slots_with_info = [
                {
                    "name": q.collect,
                    "description": q.description,
                    "allowed_values": allowed_values_for_slot(tracker.slots[q.collect]),
                }
                for q in flow.get_collect_steps()
                if self.is_extractable(q, tracker)
            ]
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

    @classmethod
    def fingerprint_addon(cls, config: Dict[str, Any]) -> Optional[str]:
        """Add a fingerprint of the knowledge base for the graph."""
        prompt_template = get_prompt_template(
            config.get("prompt"),
            DEFAULT_COMMAND_PROMPT_TEMPLATE,
        )
        return deep_container_fingerprint(prompt_template)
