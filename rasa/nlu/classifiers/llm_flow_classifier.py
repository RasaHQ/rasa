import importlib.resources
import re
import logging
from typing import Dict, Any, Optional, List, Tuple

from jinja2 import Template

from rasa.core.policies.flow_policy import FlowStack
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.core.flows.flow import FlowsList, QuestionFlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import (
    INTENT,
    EXTRACTOR,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITIES,
    TEXT,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_TEXT,
    ENTITY_ATTRIBUTE_CONFIDENCE,
    CORRECTION_INTENT,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.llm import (
    tracker_as_readable_transcript,
    generate_text_openai_chat,
    sanitize_message_for_prompt,
)

PROMPT_TEMPLATE = Template(
    importlib.resources.read_text("rasa.nlu.classifiers", "flow_prompt_template.jinja2")
)

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER,
        DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR,
    ],
    is_trainable=True,
)
class LLMFlowClassifier(GraphComponent, IntentClassifier, EntityExtractorMixin):
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {}

    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        self.config = {**self.get_default_config(), **config}
        self._model_storage = model_storage
        self._resource = resource

    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "LLMFlowClassifier":
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
    ) -> "LLMFlowClassifier":
        """Loads trained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    def train(self, training_data: TrainingData) -> Resource:
        """Train the intent classifier on a data set."""
        self.persist()
        return self._resource

    def process(
        self,
        messages: List[Message],
        tracker: Optional[DialogueStateTracker] = None,
        flows: Optional[FlowsList] = None,
    ) -> List[Message]:
        """Return intent and entities for a message."""
        return [self.process_single(msg, tracker, flows) for msg in messages]

    def process_single(
        self,
        message: Message,
        tracker: Optional[DialogueStateTracker] = None,
        flows: Optional[FlowsList] = None,
    ) -> Message:
        if flows is None or tracker is None:
            # cannot do anything if there are no flows or no tracker
            return message
        flows_without_patterns = FlowsList(
            [f for f in flows.underlying_flows if not f.is_handling_pattern()]
        )
        flow_prompt = self.render_template(message, tracker, flows_without_patterns)
        logger.info(flow_prompt)
        action_list = generate_text_openai_chat(flow_prompt)
        logger.info(action_list)
        intent_name, entities = self.parse_action_list(
            action_list, tracker, flows_without_patterns
        )
        intent = {"name": intent_name, "confidence": 0.90}
        message.set(INTENT, intent, add_to_output=True)
        if len(entities) > 0:
            formatted_entities = [
                {
                    ENTITY_ATTRIBUTE_START: 0,
                    ENTITY_ATTRIBUTE_END: 0,
                    ENTITY_ATTRIBUTE_TYPE: e[0],
                    ENTITY_ATTRIBUTE_VALUE: e[1],
                    ENTITY_ATTRIBUTE_TEXT: e[1],
                    ENTITY_ATTRIBUTE_CONFIDENCE: 0.9,
                    EXTRACTOR: self.__class__.__name__,
                }
                for e in entities
            ]
            message.set(ENTITIES, formatted_entities, add_to_output=True)
        return message

    @classmethod
    def parse_action_list(
        cls, actions: str, tracker: DialogueStateTracker, flows: FlowsList
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Parse the actions returned by the llm into intent and entities."""
        start_flow_actions = []
        slot_sets = []
        cancel_flow = False
        slot_set_re = re.compile(
            r"""SetSlot\(([a-zA-Z_][a-zA-Z0-9_-]*?), ?\"?([^)]*?)\"?\)"""
        )
        start_flow_re = re.compile(r"StartFlow\(([a-zA-Z_][a-zA-Z0-9_-]*?)\)")
        cancel_flow_re = re.compile(r"CancelFlow")
        for action in actions.strip().splitlines():
            if m := slot_set_re.search(action):
                slot_name = m.group(1).strip()
                slot_value = m.group(2).strip()
                if slot_value == "undefined":
                    continue
                if slot_name == "flow_name":
                    start_flow_actions.append(slot_value)
                else:
                    # most likely some hallucinated variable
                    if "_" in slot_value:
                        continue
                    slot_sets.append((slot_name, slot_value))
            elif m := start_flow_re.search(action):
                start_flow_actions.append(m.group(1).strip())
            elif cancel_flow_re.search(action):
                cancel_flow = True

        # case 1
        # "I want to send some money"
        # starting a flow -> intent = flow name

        # case 2
        # "I want to send some money to Joe"
        # starting a flow with entities mentioned -> intent = flow name, entities only those that are valid for the flow

        # case 3
        # "50$"
        # giving information for the current slot -> intent = inform, entity only that of the current slot

        # case 4
        # "Sorry I meant, Joe, not John"
        # correcting a previous slot from the flow -> intent = correction, entity of the previous slot

        # everything else is too complex for now:
        # case 5
        # "50$, how much money do I still have btw?"
        # giving information about current flow and starting new flow right away -> intent = complex

        # TODO: check that we have a valid flow name if any, reprompt if mistake?
        # TODO: assign slot sets to current flow, new flow if any, and other

        flow_stack = FlowStack.from_tracker(tracker)
        top_flow = flow_stack.top_flow(flows)
        top_flow_step = flow_stack.top_flow_step(flows)
        if top_flow_step is not None and top_flow is not None:
            slots_so_far = top_flow.slots_up_to_step(top_flow_step.id)
            other_slots = [
                slot_set for slot_set in slot_sets if slot_set[0] not in slots_so_far
            ]
        else:
            slots_so_far = []
            other_slots = slot_sets

        if len(start_flow_actions) == 0:
            if len(slot_sets) == 0 and not cancel_flow:
                return "comment", []
            elif len(slot_sets) == 0 and cancel_flow:
                return "cancel_flow", []
            elif (
                len(slot_sets) == 1
                and isinstance(top_flow_step, QuestionFlowStep)
                and top_flow_step.question == slot_sets[0][0]
            ):
                return "inform", slot_sets
            elif (
                len(slot_sets) == 1
                and isinstance(top_flow_step, QuestionFlowStep)
                and top_flow_step.question != slot_sets[0][0]
                and slot_sets[0][0] in slots_so_far
            ):
                return CORRECTION_INTENT, slot_sets
            elif (
                len(slot_sets) == 1
                and top_flow_step is not None
                and slot_sets[0][0] in other_slots
            ):
                # trying to set a slot from another flow
                return "too_complex", []
            elif len(slot_sets) > 1:
                return "too_complex", []
        elif len(start_flow_actions) == 1:
            if cancel_flow:
                return "too_complex", []
            new_flow_id = start_flow_actions[0]
            potential_new_flow = flows.flow_by_id(new_flow_id)
            if potential_new_flow is not None:
                valid_slot_sets = [
                    slot_set
                    for slot_set in slot_sets
                    if slot_set[0] in potential_new_flow.slots()
                ]
                return start_flow_actions[0], valid_slot_sets
            else:
                return "mistake", []
                # TODO: potentially re-prompt or ask for correction on invalid flow name
        elif len(start_flow_actions) > 1:
            return "too_complex", []

        return "too_complex", []

    @classmethod
    def create_template_inputs(cls, flows: FlowsList) -> List[Dict[str, Any]]:
        result = []
        for flow in flows.underlying_flows:
            result.append(
                {
                    "name": flow.id,
                    "description": flow.description,
                    "slots": flow.slots(),
                }
            )
        return result

    @classmethod
    def render_template(
        cls, message: Message, tracker: DialogueStateTracker, flows: FlowsList
    ) -> str:
        flow_stack = FlowStack.from_tracker(tracker)
        top_flow = flow_stack.top_flow(flows) if flow_stack is not None else None
        current_step = (
            flow_stack.top_flow_step(flows) if flow_stack is not None else None
        )
        if top_flow is not None:
            flow_slots = [
                {
                    "name": k,
                    "value": (tracker.get_slot(k) or "undefined"),
                    "type": tracker.slots[k].type_name,
                }
                for k in top_flow.slots()
            ]
        else:
            flow_slots = []

        question = (
            current_step.question
            if current_step is not None and isinstance(current_step, QuestionFlowStep)
            else None
        )
        current_conversation = tracker_as_readable_transcript(tracker)
        latest_user_message = sanitize_message_for_prompt(message.get(TEXT))
        current_conversation += f"\nUSER: {latest_user_message}"

        inputs = {
            "available_flows": cls.create_template_inputs(flows),
            "current_conversation": current_conversation,
            "flow_slots": flow_slots,
            "current_flow": top_flow.id if top_flow is not None else None,
            "question": question,
            "user_message": latest_user_message,
        }

        return PROMPT_TEMPLATE.render(**inputs)
