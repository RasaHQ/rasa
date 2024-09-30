import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Text, Tuple, Union

import tiktoken
from numpy import ndarray
from rasa_sdk.grpc_py import action_webhook_pb2

from rasa.core.actions.action import DirectCustomActionExecutor
from rasa.core.actions.grpc_custom_action_executor import GRPCCustomActionExecutor
from rasa.core.actions.http_custom_action_executor import HTTPCustomActionExecutor
from rasa.core.agent import Agent
from rasa.core.brokers.broker import EventBroker
from rasa.core.channels import UserMessage
from rasa.core.lock_store import LOCK_LIFETIME, LockStore
from rasa.core.nlg.contextual_response_rephraser import ContextualResponseRephraser
from rasa.core.processor import MessageProcessor
from rasa.core.tracker_store import TrackerStore
from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.engine.graph import ExecutionContext, GraphModelConfiguration, GraphNode
from rasa.engine.training.graph_trainer import GraphTrainer
from rasa.shared.constants import (
    EMBEDDINGS_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    TIMEOUT_CONFIG_KEY,
    DEPLOYMENT_CONFIG_KEY,
)
from rasa.shared.core.constants import REQUESTED_SLOT
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import DialogueStackUpdated, Event
from rasa.shared.core.flows import Flow, FlowsList, FlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.constants import INTENT_NAME_KEY, SET_SLOT_COMMAND
from rasa.shared.utils.llm import (
    combine_custom_and_default_config,
)
from rasa.tracing.constants import (
    PROMPT_TOKEN_LENGTH_ATTRIBUTE_NAME,
    REQUEST_BODY_SIZE_IN_BYTES_ATTRIBUTE_NAME,
)
from rasa.shared.core.training_data.structures import StoryGraph

if TYPE_CHECKING:
    from langchain.llms.base import BaseLLM

    from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy
    from rasa.core.policies.intentless_policy import IntentlessPolicy
    from rasa.core.policies.policy import PolicyPrediction
    from rasa.dialogue_understanding.generator import (
        CommandGenerator,
        LLMBasedCommandGenerator,
    )

# This file contains all attribute extractors for tracing instrumentation.
# These are functions that are applied to the arguments of the wrapped function to be
# traced to extract the attributes that we want to forward to our tracing backend.
# Note that we always mirror the argument lists of the wrapped functions, as our
# wrapping mechanism always passes in the original arguments unchanged for further
# processing.

logger = logging.getLogger(__name__)


def extract_attrs_for_agent(
    self: Agent,
    message: UserMessage,
) -> Dict[str, Any]:
    """Extract the attributes for `Agent.handle_message`.

    :param self: The `Agent` on which `handle_message` is called.
    :param message: The `UserMessage` argument.
    :return: A dictionary containing the attributes.
    """
    return {
        "input_channel": str(message.input_channel),
        "sender_id": message.sender_id,
        "model_id": str(self.model_id),
        "model_name": self.processor.model_filename if self.processor else "None",
    }


def extract_llm_command_generator_attrs(
    attributes: Dict[str, Any], commands: List[Dict[str, Any]]
) -> None:
    """Extract more attributes for `GraphNode` type `LLMCommandGenerator`.

    :param attributes: A dictionary containing attributes.
    :param commands: The commands to execute.
    """
    commands_list = []

    for command in commands:
        command_name = command.get("command")
        commands_list.append(command_name)

        if command_name == SET_SLOT_COMMAND:
            attributes["slot_name"] = command.get("name")

        if command_name == "start flow":
            attributes["flow_name"] = command.get("flow")

    attributes["commands"] = str(commands_list)


def extract_flow_policy_attrs(
    attributes: Dict[str, Any], policy_prediction: "PolicyPrediction"
) -> None:
    """Extract more attributes for `GraphNode` type `FlowPolicy`.

    :param attributes: A dictionary containing attributes.
    :param policy_prediction: The PolicyPrediction to use.
    """
    attributes["policy"] = policy_prediction.policy_name

    if policy_prediction.events:
        attributes["events"] = str(
            [event.__class__.__name__ for event in policy_prediction.events]
        )

    if policy_prediction.optional_events:
        optional_events_name = []
        flows = []
        utters = []

        for optional_event in policy_prediction.optional_events:
            optional_events_name.append(optional_event.__class__.__name__)

            if (
                isinstance(optional_event, DialogueStackUpdated)
                and "value" in optional_event.update
            ):
                updates = json.loads(optional_event.update)
                for update in updates:
                    value = update.get("value", {})
                    if isinstance(value, dict):
                        flow = value.get("flow_id", "")
                        utter = value.get("utter", "")
                        if flow:
                            flows.append(flow)
                        if utter:
                            utters.append(utter)
            else:
                if hasattr(optional_event, "flow_id") and optional_event.flow_id:
                    flows.append(optional_event.flow_id)
                if hasattr(optional_event, "utter") and optional_event.utter:
                    utters.append(optional_event.utter)

        attributes["optional_events"] = str(set(optional_events_name))

        if flows:
            attributes["flows"] = str(set(flows))
        if utters:
            attributes["utters"] = str(set(utters))


def extract_attrs_for_graph_node(
    self: GraphNode, *inputs_from_previous_nodes: Tuple[Text, Any]
) -> Dict[str, Any]:
    """Extract the attributes for `GraphNode.__call__`.

    :param self: The `GraphNode` on which `__call__` is called.
    :param inputs_from_previous_nodes: Unused outputs of all parent nodes.
    :return: A dictionary containing the attributes.
    """
    attributes = {
        "node_name": self._node_name,
        "component_class": self._component_class.__name__,
        "fn_name": self._fn_name,
    }

    for node_input in inputs_from_previous_nodes:
        if "LLMCommandGenerator" in node_input[0]:
            commands = node_input[1][0].data.get("commands")
            extract_llm_command_generator_attrs(attributes, commands)

        if "FlowPolicy" in node_input[0]:
            policy_prediction = node_input[1]
            extract_flow_policy_attrs(attributes, policy_prediction)

    return attributes


def extract_number_of_events(
    self: MessageProcessor, tracker: DialogueStateTracker
) -> Dict[str, Any]:
    """Extract the attributes for `MessageProcessor.save_tracker`.

    :param self: The `MessageProcessor` on which `save_tracker` is called.
    :param tracker: The `DialogueStateTracker` argument.
    :return: A dictionary containing the attributes.
    """
    return {"number_of_events": len(tracker.events)}


def extract_attrs_for_tracker_store(
    self: TrackerStore,
    event_broker: EventBroker,
    new_events: List[Event],
    sender_id: Text,
) -> Dict[str, Any]:
    """Extract the attributes for `TrackerStore.stream_events`.

    :param self: The `TrackerStore` on which `stream_events` is called.
    :param event_broker: The `EventBroker` on which the new events are published.
    :param new_events: List of new events to stream.
    :param sender_id: The sender id of the tracker to which the new events were added.
    """
    return {
        "number_of_streamed_events": len(new_events),
        "broker_class": self.event_broker.__class__.__name__,
    }


def extract_attrs_for_lock_store(
    self: LockStore,
    conversation_id: Text,
    lock_lifetime: float = LOCK_LIFETIME,
    wait_time_in_seconds: float = 1,
) -> Dict[str, Any]:
    """Extract the attributes for `LockStore.lock`.

    :param self: the `LockStore` on which `lock` is called.
    :param conversation_id: The conversation id for which the lock is acquired.
    :param lock_lifetime: The lifetime of the lock.
    :param wait_time_in_seconds: The time to wait for the lock.
    :return: A dictionary containing the attributes.
    """
    return {"lock_store_class": self.__class__.__name__}


def extract_attrs_for_graph_trainer(
    self: GraphTrainer,
    model_configuration: GraphModelConfiguration,
    importer: TrainingDataImporter,
    output_filename: Path,
    is_finetuning: bool = False,
    force_retraining: bool = False,
) -> Dict[str, Any]:
    """Extract the attributes for `GraphTrainer.train`.

    :param self: the `GraphTrainer` on which `train` is called.
    :param model_configuration: The model configuration (training_type, language etc.).
    :param importer: The importer which provides the training data for the training.
    :param output_filename: The location where the packaged model is saved.
    :param is_finetuning: Boolean argument, if `True` enables incremental training.
    :param force_retraining: Unused boolean argument,i.e, if `True` then the cache
    is skipped and all components are retrained.
    :return: A dictionary containing the attributes.
    """
    return {
        "training_type": model_configuration.training_type.model_type,
        "language": model_configuration.language,
        "recipe_name": importer.get_config().get("recipe"),
        "output_filename": output_filename.name,
        "is_finetuning": is_finetuning,
    }


def extract_headers(message: UserMessage, **kwargs: Any) -> Any:
    """Extract the headers from the `UserMessage`."""
    if message.headers:
        return message.headers
    return {}


def extract_intent_name_and_slots(
    self: MessageProcessor, tracker: DialogueStateTracker
) -> Dict[str, Any]:
    """Extract the attributes for `MessageProcessor._predict_next_with_tracker`.

    :param self: The `MessageProcessor` on which `_predict_next_with_tracker` is called.
    :param tracker: The `DialogueStateTracker` argument.
    :return: A dictionary containing the attributes.
    """
    slots = {}
    for slot_name, slot_value in tracker.slots.items():
        if slot_name == REQUESTED_SLOT and slot_value.value:
            slots[slot_name] = slot_value.value
            break
    return {
        "intent_name": str(tracker.latest_message.intent.get(INTENT_NAME_KEY)),  # type: ignore[union-attr]
        **slots,
    }


def extract_attrs_for_command(
    self: Command,
    tracker: DialogueStateTracker,
    all_flows: FlowsList,
    original_tracker: DialogueStateTracker,
) -> Dict[str, Any]:
    return {
        "class_name": self.__class__.__name__,
        "number_of_events": len(tracker.events),
        "sender_id": tracker.sender_id,
    }


def extract_llm_config(self: Any, default_llm_config: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(self, ContextualResponseRephraser):
        config = self.nlg_endpoint.kwargs
    else:
        config = self.config

    llm_property = combine_custom_and_default_config(
        config.get("llm"), default_llm_config
    )

    attributes = {
        "class_name": self.__class__.__name__,
        "llm_model": str(llm_property.get(MODEL_CONFIG_KEY)),
        "llm_type": str(llm_property.get(PROVIDER_CONFIG_KEY)),
        "embeddings": json.dumps(config.get(EMBEDDINGS_CONFIG_KEY, {})),
        "llm_temperature": str(llm_property.get("temperature")),
        "request_timeout": str(llm_property.get(TIMEOUT_CONFIG_KEY)),
    }

    if DEPLOYMENT_CONFIG_KEY in llm_property:
        attributes["llm_engine"] = str(llm_property.get(DEPLOYMENT_CONFIG_KEY))

    return attributes


def extract_attrs_for_llm_based_command_generator(
    self: "LLMBasedCommandGenerator",
    prompt: str,
) -> Dict[str, Any]:
    from rasa.dialogue_understanding.generator.constants import (
        DEFAULT_LLM_CONFIG,
    )

    attributes = extract_llm_config(self, default_llm_config=DEFAULT_LLM_CONFIG)

    return extend_attributes_with_prompt_tokens_length(self, attributes, prompt)


def extract_attrs_for_contextual_response_rephraser(
    self: Any,
    prompt: str,
) -> Dict[str, Any]:
    from rasa.core.nlg.contextual_response_rephraser import DEFAULT_LLM_CONFIG

    attributes = extract_llm_config(self, default_llm_config=DEFAULT_LLM_CONFIG)

    return extend_attributes_with_prompt_tokens_length(self, attributes, prompt)


def extract_attrs_for_create_history(
    self: Any,
    tracker: DialogueStateTracker,
) -> Dict[str, Any]:
    from rasa.core.nlg.contextual_response_rephraser import DEFAULT_LLM_CONFIG

    return extract_llm_config(self, default_llm_config=DEFAULT_LLM_CONFIG)


def extract_attrs_for_generate(
    self: Any,
    utter_action: Text,
    tracker: DialogueStateTracker,
    output_channel: Text,
    **kwargs: Any,
) -> Optional[Dict[Text, Any]]:
    return {
        "class_name": self.__class__.__name__,
        "utter": utter_action,
    }


def extract_attrs_for_execute_commands(
    tracker: DialogueStateTracker,
    all_flows: FlowsList,
    execution_context: ExecutionContext,
    story_graph: Optional[StoryGraph] = None,
) -> Dict[str, Any]:
    return {
        "number_of_events": len(tracker.events),
        "sender_id": tracker.sender_id,
        "module_name": "command_processor",
    }


def extract_attrs_for_validate_state_of_commands(
    commands: List[Command],
) -> Dict[str, Any]:
    commands_list = []

    for command in commands:
        command_type = command.command()
        command_as_dict = command.as_dict()

        if command_type == SET_SLOT_COMMAND:
            command_as_dict.pop("value", None)

        if command_type == "correct slot":
            corrected_slots = command_as_dict.get("corrected_slots", [])
            updated_corrected_slots = []
            for corrected_slot in corrected_slots:
                corrected_slot.pop("value", None)
                updated_corrected_slots.append(corrected_slot)

            command_as_dict["corrected_slots"] = updated_corrected_slots

        commands_list.append(command_as_dict)

    return {
        "cleaned_up_commands": str(commands_list),
        "module_name": "command_processor",
    }


def extract_attrs_for_clean_up_commands(
    commands: List[Command],
    tracker: DialogueStateTracker,
    all_flows: FlowsList,
    execution_context: ExecutionContext,
    story_graph: Optional[StoryGraph] = None,
) -> Dict[str, Any]:
    commands_list = []

    for command in commands:
        command_type = command.command()
        command_as_dict = command.as_dict()

        if command_type == SET_SLOT_COMMAND:
            command_as_dict.pop("value", None)

        commands_list.append(command_as_dict)

    current_context = extract_current_context_attribute(tracker.stack)

    return {
        "commands": str(commands_list),
        "module_name": "command_processor",
        "current_context": json.dumps(current_context),
    }


def extract_attrs_for_remove_duplicated_set_slots(
    events: List[Event],
) -> Dict[str, Any]:
    resulting_events = []

    for event in events:
        event_as_dict = event.as_dict()

        if event_as_dict.get("event") == "stack":
            update = event_as_dict.pop("update", "")
            if update:
                update = json.loads(update)
                for update_dict in update:
                    value = update_dict.pop("value", {})
                    value.pop("corrected_slots", None)
                    update_dict["value"] = json.dumps(value)
                    event_as_dict["update"] = str([update_dict])
                    break

        elif event_as_dict.get("event") == "slot":
            event_as_dict.pop("value", None)

        resulting_events.append(event_as_dict)

    return {
        "resulting_events": str(resulting_events),
        "module_name": "command_processor",
    }


def extract_attrs_for_check_commands_against_startable_flows(
    self: "CommandGenerator", commands: List[Command], startable_flows: FlowsList
) -> Dict[str, Any]:
    commands_list = []

    for command in commands:
        command_as_dict = command.as_dict()
        command_type = command.command()

        if command_type == SET_SLOT_COMMAND:
            slot_value = command_as_dict.pop("value", None)
            command_as_dict["is_slot_value_missing_or_none"] = slot_value is None

        commands_list.append(command_as_dict)

    startable_flow_ids = [flow.id for flow in startable_flows.underlying_flows]

    return {
        "commands": json.dumps(commands_list),
        "startable_flow_ids": json.dumps(startable_flow_ids),
    }


def extract_attrs_for_advance_flows(
    tracker: DialogueStateTracker, available_actions: List[str], flows: FlowsList
) -> Dict[str, Any]:
    from rasa.tracing.instrumentation.instrumentation import FLOW_EXECUTOR_MODULE_NAME

    current_context = extract_current_context_attribute(tracker.stack)

    return {
        "module_name": FLOW_EXECUTOR_MODULE_NAME,
        "available_actions": json.dumps(available_actions),
        "current_context": json.dumps(current_context),
    }


def extract_attrs_for_run_step(
    step: FlowStep,
    flow: Flow,
    stack: DialogueStack,
    tracker: DialogueStateTracker,
    available_actions: List[str],
    flows: FlowsList,
) -> Dict[str, Any]:
    current_context = extract_current_context_attribute(stack)

    return {
        "step_custom_id": step.custom_id if step.custom_id else "None",
        "step_description": step.description if step.description else "None",
        "current_flow_id": flow.id,
        "current_context": json.dumps(current_context),
    }


def extract_attrs_for_policy_prediction(
    self: Any,
    probabilities: List[float],
    events: Optional[List[Event]] = None,
    optional_events: Optional[List[Event]] = None,
    is_end_to_end_prediction: bool = False,
    is_no_user_prediction: bool = False,
    diagnostic_data: Optional[Dict[Text, Any]] = None,
    action_metadata: Optional[Dict[Text, Any]] = None,
) -> Dict[str, Any]:
    # diagnostic_data can contain ndarray type values which need to be converted
    # into a list since the returning values have to be JSON serializable.
    if isinstance(diagnostic_data, dict):
        diagnostic_data = {
            key: value.tolist() if isinstance(value, ndarray) else value
            for key, value in diagnostic_data.items()
        }

    return {
        "priority": self.priority,
        "events": [event.__class__.__name__ for event in events] if events else "None",
        "optional_events": [event.__class__.__name__ for event in optional_events]
        if optional_events
        else "None",
        "is_end_to_end_prediction": is_end_to_end_prediction,
        "is_no_user_prediction": is_no_user_prediction,
        "diagnostic_data": json.dumps(diagnostic_data),
        "action_metadata": json.dumps(action_metadata),
    }


def extract_attrs_for_intentless_policy_prediction_result(
    self: "IntentlessPolicy",
    action_name: Optional[Text],
    domain: Domain,
    score: Optional[float] = 1.0,
) -> Dict[str, Any]:
    return {
        "action_name": action_name if action_name else "null",
        "score": score if score else 0.0,
    }


def extract_attrs_for_intentless_policy_find_closest_response(
    self: "IntentlessPolicy",
    tracker: DialogueStateTracker,
) -> Dict[str, Any]:
    return {
        "current_context": json.dumps(tracker.stack.current_context()),
    }


def extract_attrs_for_intentless_policy_generate_llm_answer(
    self: "IntentlessPolicy", llm: "BaseLLM", prompt: str
) -> Dict[str, Any]:
    from rasa.core.policies.intentless_policy import DEFAULT_LLM_CONFIG

    attributes = extract_llm_config(self, default_llm_config=DEFAULT_LLM_CONFIG)

    return extend_attributes_with_prompt_tokens_length(self, attributes, prompt)


def extract_attrs_for_enterprise_search_generate_llm_answer(
    self: "EnterpriseSearchPolicy", llm: "BaseLLM", prompt: str
) -> Dict[str, Any]:
    from rasa.core.policies.enterprise_search_policy import DEFAULT_LLM_CONFIG

    attributes = extract_llm_config(self, default_llm_config=DEFAULT_LLM_CONFIG)

    return extend_attributes_with_prompt_tokens_length(self, attributes, prompt)


def extract_current_context_attribute(stack: DialogueStack) -> Dict[str, Any]:
    """Utility function to extract the current context from the dialogue stack."""
    current_context = stack.current_context()

    if "corrected_slots" in current_context:
        current_context["corrected_slots"] = list(
            current_context["corrected_slots"].keys()
        )

    return current_context


def compute_prompt_tokens_length(
    model_type: str, model_name: str, prompt: str
) -> Optional[int]:
    """Utility function to compute the length of the prompt tokens for OpenAI models."""
    if model_type != "openai":
        logger.warning(
            "Tracing prompt tokens is only supported for OpenAI models. Skipping."
        )
        return None

    if model_name in ["gpt-3.5-turbo", "gpt-4"]:
        logger.debug(
            f"Model {model_name} may update over time. "
            f"Returning num tokens assuming model '{model_name}-0613.'"
        )
        model_name = f"{model_name}-0613"

    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(prompt))


def extend_attributes_with_prompt_tokens_length(
    self: Any,
    attributes: Dict[str, Any],
    prompt: str,
) -> Dict[str, Any]:
    if not self.trace_prompt_tokens:
        return attributes

    len_prompt_tokens = compute_prompt_tokens_length(
        model_type=attributes["llm_type"],
        model_name=attributes["llm_model"],
        prompt=prompt,
    )

    attributes[PROMPT_TOKEN_LENGTH_ATTRIBUTE_NAME] = str(len_prompt_tokens)

    return attributes


def extract_attrs_for_custom_action_executor_run(
    self: Union[
        HTTPCustomActionExecutor, GRPCCustomActionExecutor, DirectCustomActionExecutor
    ],
    tracker: DialogueStateTracker,
    domain: Domain,
    include_domain: bool = False,
) -> Dict[str, Any]:
    actions_module, url = None, None
    if hasattr(self, "action_endpoint"):
        url = self.action_endpoint.url
        actions_module = self.action_endpoint.actions_module

    attrs: Dict[str, Any] = {
        "class_name": self.__class__.__name__,
        "action_name": self.action_name,
        "sender_id": tracker.sender_id,
        "url": str(url),
        "actions_module": str(actions_module),
    }
    return attrs


def extract_attrs_for_grpc_custom_action_executor_request(
    self: GRPCCustomActionExecutor,
    request: action_webhook_pb2.WebhookRequest,
) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {"url": self.action_endpoint.url}

    attrs.update(
        {
            REQUEST_BODY_SIZE_IN_BYTES_ATTRIBUTE_NAME: request.ByteSize(),
        }
    )

    return attrs
