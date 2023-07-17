import importlib.resources
import re
from typing import Dict, Any, Optional, List

from jinja2 import Template
import structlog
from rasa.cdu.classifiers.base import CommandClassifier
from rasa.cdu.commands import (
    Command,
    SetSlotCommand,
    CancelFlowCommand,
    StartFlowCommand,
)

from rasa.core.policies.flow_policy import FlowStack
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.constants import (
    MAPPING_TYPE,
    SlotMappingType,
    MAPPING_CONDITIONS,
    ACTIVE_LOOP,
)
from rasa.shared.core.flows.flow import FlowsList, QuestionFlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import (
    TEXT,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.llm import (
    DEFAULT_OPENAI_CHAT_MODEL_NAME,
    llm_factory,
    tracker_as_readable_transcript,
    sanitize_message_for_prompt,
)

DEFAULT_FLOW_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.nlu.classifiers", "flow_prompt_template.jinja2"
)

structlogger = structlog.get_logger()


DEFAULT_LLM_CONFIG = {
    "_type": "openai",
    "request_timeout": 5,
    "temperature": 0.0,
    "model_name": DEFAULT_OPENAI_CHAT_MODEL_NAME,
}

LLM_CONFIG_KEY = "llm"


# TODO: check if the original inhertance from IntentClassifier and EntityExtractorMixin
#   is still needed or what benefits that provided.
@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER,
        DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR,
    ],
    is_trainable=True,
)
class LLMCommandClassifier(GraphComponent, CommandClassifier):
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            "prompt": DEFAULT_FLOW_PROMPT_TEMPLATE,
            LLM_CONFIG_KEY: None,
        }

    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        self.config = {**self.get_default_config(), **config}
        self.prompt_template = self.config["prompt"]
        self._model_storage = model_storage
        self._resource = resource

    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "LLMCommandClassifier":
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    def persist(self) -> None:
        pass

    @classmethod
    def load(
        cls,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> "LLMCommandClassifier":
        """Loads trained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    def train(self, training_data: TrainingData) -> Resource:
        """Train the intent classifier on a data set."""
        self.persist()
        return self._resource

    def _generate_action_list_using_llm(self, prompt: str) -> Optional[str]:
        """Use LLM to generate a response.

        Args:
            prompt: the prompt to send to the LLM

        Returns:
            generated text
        """
        llm = llm_factory(self.config.get(LLM_CONFIG_KEY), DEFAULT_LLM_CONFIG)

        try:
            return llm(prompt)
        except Exception as e:
            # unfortunately, langchain does not wrap LLM exceptions which means
            # we have to catch all exceptions here
            structlogger.error("llm_command_classifier.llm.error", error=e)
            return None

    def predict_commands(
        self,
        message: Message,
        tracker: Optional[DialogueStateTracker] = None,
        flows: Optional[FlowsList] = None,
    ) -> List[Command]:
        if flows is None or tracker is None:
            # cannot do anything if there are no flows or no tracker
            return []
        flows_without_patterns = FlowsList(
            [f for f in flows.underlying_flows if not f.is_handling_pattern()]
        )
        flow_prompt = self.render_template(message, tracker, flows_without_patterns)
        structlogger.info(
            "llm_command_classifier.process.prompt_rendered", prompt=flow_prompt
        )
        action_list = self._generate_action_list_using_llm(flow_prompt)
        structlogger.info(
            "llm_command_classifier.process.actions_generated", action_list=action_list
        )
        commands = self.parse_commands(action_list, tracker, flows_without_patterns)
        structlogger.info(
            "llm_command_classifier.process.finished",
            commands=commands,
        )
        return commands

    @staticmethod
    def is_hallucinated_value(value: str) -> bool:
        return "_" in value or value in {
            "[missing information]",
            "[missing]",
            "None",
            "undefined",
        }

    @classmethod
    def parse_commands(
        cls, actions: Optional[str], tracker: DialogueStateTracker, flows: FlowsList
    ) -> List[Command]:
        """Parse the actions returned by the llm into intent and entities."""
        if not actions:
            # TODO: not quite sure yet how to handle this case - revisit!
            #  is predicting "no commands" an option?
            return []

        commands: List[Command] = []

        slot_set_re = re.compile(
            r"""SetSlot\(([a-zA-Z_][a-zA-Z0-9_-]*?), ?\"?([^)]*?)\"?\)"""
        )
        start_flow_re = re.compile(r"StartFlow\(([a-zA-Z_][a-zA-Z0-9_-]*?)\)")
        cancel_flow_re = re.compile(r"CancelFlow")
        for action in actions.strip().splitlines():
            if m := slot_set_re.search(action):
                slot_name = m.group(1).strip()
                slot_value = m.group(2).strip()
                if slot_name == "flow_name":
                    commands.append(StartFlowCommand(flow=slot_value))
                elif cls.is_hallucinated_value(slot_value):
                    continue
                else:
                    commands.append(SetSlotCommand(name=slot_name, value=slot_value))
            elif m := start_flow_re.search(action):
                commands.append(StartFlowCommand(flow=m.group(1).strip()))
            elif cancel_flow_re.search(action):
                commands.append(CancelFlowCommand())

        return commands

    @classmethod
    def create_template_inputs(
        cls, flows: FlowsList, tracker: DialogueStateTracker
    ) -> List[Dict[str, Any]]:
        result = []
        for flow in flows.underlying_flows:
            if not flow.is_rasa_default_flow():

                slots_with_info = [
                    {"name": q.question, "description": q.description}
                    for q in flow.get_question_steps()
                    if cls.is_extractable(q, tracker)
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
    def is_extractable(q: QuestionFlowStep, tracker: DialogueStateTracker) -> bool:
        slot = tracker.slots.get(q.question)
        if slot is None:
            return False

        for mapping in slot.mappings:
            if mapping.get(MAPPING_TYPE) == str(SlotMappingType.FROM_ENTITY):
                conditions = mapping.get(MAPPING_CONDITIONS, [])
                if len(conditions) == 0:
                    return True
                else:
                    for condition in conditions:
                        active_loop = condition.get(ACTIVE_LOOP)
                        if active_loop and active_loop == tracker.active_loop_name:
                            return True
        return False

    def render_template(
        self, message: Message, tracker: DialogueStateTracker, flows: FlowsList
    ) -> str:
        flow_stack = FlowStack.from_tracker(tracker)
        top_flow = flow_stack.top_flow(flows) if flow_stack is not None else None
        current_step = (
            flow_stack.top_flow_step(flows) if flow_stack is not None else None
        )
        if top_flow is not None:
            flow_slots = [
                {
                    "name": q.question,
                    "value": (tracker.get_slot(q.question) or "undefined"),
                    "type": tracker.slots[q.question].type_name,
                    "description": q.description,
                }
                for q in top_flow.get_question_steps()
                if self.is_extractable(q, tracker)
            ]
        else:
            flow_slots = []

        question, question_description = (
            (current_step.question, current_step.description)
            if isinstance(current_step, QuestionFlowStep)
            else (None, None)
        )
        current_conversation = tracker_as_readable_transcript(tracker)
        latest_user_message = sanitize_message_for_prompt(message.get(TEXT))
        current_conversation += f"\nUSER: {latest_user_message}"

        inputs = {
            "available_flows": self.create_template_inputs(flows, tracker),
            "current_conversation": current_conversation,
            "flow_slots": flow_slots,
            "current_flow": top_flow.id if top_flow is not None else None,
            "question": question,
            "question_description": question_description,
            "user_message": latest_user_message,
        }

        return Template(self.prompt_template).render(**inputs)
