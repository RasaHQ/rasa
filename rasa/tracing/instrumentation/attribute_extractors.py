import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Text, Tuple

from rasa.core.agent import Agent
from rasa.core.brokers.broker import EventBroker
from rasa.core.channels import UserMessage
from rasa.core.lock_store import LOCK_LIFETIME, LockStore
from rasa.core.processor import MessageProcessor
from rasa.core.tracker_store import TrackerStore
from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.generator.llm_command_generator import (
    LLMCommandGenerator,
)
from rasa.engine.graph import GraphModelConfiguration, GraphNode
from rasa.engine.training.graph_trainer import GraphTrainer
from rasa.shared.core.constants import REQUESTED_SLOT
from rasa.shared.core.events import DialogueStackUpdated, Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.constants import INTENT_NAME_KEY

if TYPE_CHECKING:
    from rasa.core.policies.flow_policy import FlowPolicy
    from rasa.dialogue_understanding.generator.command_generator import CommandGenerator

# This file contains all attribute extractors for tracing instrumentation.
# These are functions that are applied to the arguments of the wrapped function to be
# traced to extract the attributes that we want to forward to our tracing backend.
# Note that we always mirror the argument lists of the wrapped functions, as our
# wrapping mechanism always passes in the original arguments unchanged for further
# processing.


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
        "input_channel": message.input_channel,
        "sender_id": message.sender_id,
        "model_id": self.model_id,
        "model_name": self.model_name,
    }


def extract_llm_command_generator_attrs(
    attributes: Dict[str, Any], commands: List[Command]
) -> None:
    """Extract more attributes for `GraphNode` type `LLMCommandGenerator`.

    :param attributes: A dictionary containing attributes.
    :param commands: The commands to execute.
    """
    commands_list = []

    for command in commands:
        command_name = command.get("command")  # type: ignore[attr-defined]
        commands_list.append(command_name)

        if command_name == "set slot":
            attributes["slot_name"] = command.get("name")  # type: ignore[attr-defined]

        if command_name == "start flow":
            attributes["flow_name"] = command.get("flow")  # type: ignore[attr-defined]

    attributes["commands"] = str(commands_list)


def extract_flow_policy_attrs(
    attributes: Dict[str, Any], flow_policy: "FlowPolicy"
) -> None:
    """Extract more attributes for `GraphNode` type `FlowPolicy`.

    :param attributes: A dictionary containing attributes.
    :param commands: The FlowPolicy to use.
    """
    attributes["policy"] = flow_policy.policy_name  # type: ignore[attr-defined]

    if flow_policy.events:  # type: ignore[attr-defined]
        attributes["events"] = str(
            [event.__class__.__name__ for event in flow_policy.events]  # type: ignore[attr-defined]  # noqa: E501
        )

    if flow_policy.optional_events:  # type: ignore[attr-defined]
        optional_events_name = []
        flows = []
        utters = []

        for optional_event in flow_policy.optional_events:  # type: ignore[attr-defined]
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

    for input in inputs_from_previous_nodes:
        if "LLMCommandGenerator" in input[0]:
            commands = input[1][0].data.get("commands")
            extract_llm_command_generator_attrs(attributes, commands)

        if "FlowPolicy" in input[0]:
            extract_flow_policy_attrs(attributes, input[1])

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
    :param _event_broker: The `EventBroker` on which the new events are published.
    :param new_events: List of new events to stream.
    :param _sender_id: The sender id of the tracker to which the new events were added.
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
    :param _args: Unused additional parameters.
    :param _kwargs: Unused additional parameters.
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

    :param _graph_trainer_class: the `GraphTrainer` on which `train` is called.
    :param model_configuration: The model configuration (training_type, language etc).
    :param importer: The importer which provides the training data for the training.
    :param output_filename: The location where the packaged model is saved.
    :param is_finetuning: Boolean argument, if `True` enables incremental training.
    :param _force_retraining: Unused boolean argument,i.e, if `True` then the cache
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
        "intent_name": str(tracker.latest_message.intent.get(INTENT_NAME_KEY)),  # type: ignore[union-attr]  # noqa: E501
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


def extract_attrs_for_llm_command_generator(
    self: LLMCommandGenerator,
    prompt: str,
) -> Dict[str, Any]:
    attributes = {
        "class_name": self.__class__.__name__,
        "llm_model": str(self.config.get("model", "gpt-4")),
        "llm_type": "openai",
    }

    llm_property = self.config.get("llm") or {}

    if llm_property and ("model_name" in llm_property or "model" in llm_property):
        attributes["llm_model"] = str(
            llm_property.get("model_name") or llm_property.get("model")
        )

    attributes["llm_temperature"] = str(llm_property.get("temperature", 0.0))
    attributes["request_timeout"] = str(llm_property.get("request_timeout", 7))

    if "type" in llm_property:
        attributes["llm_type"] = str(llm_property.get("type"))

    if "engine" in llm_property:
        attributes["llm_engine"] = str(llm_property.get("engine"))

    if "deployment" in llm_property:
        attributes["llm_deployment"] = str(llm_property.get("deployment"))
    elif "deployment" in llm_property.get("embeddings", {}):
        attributes["llm_deployment"] = str(
            llm_property.get("embeddings", {}).get("deployment")
        )

    return attributes


def extract_attrs_for_contextual_response_rephraser(
    self: Any,
    prompt: str,
) -> Dict[str, Any]:
    attributes = {
        "class_name": self.__class__.__name__,
        "llm_type": self.llm_property("_type"),
    }

    if self.llm_property("model_name") or self.llm_property("model"):
        attributes["llm_model"] = self.llm_property("model_name") or self.llm_property(
            "model"
        )
    return attributes


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
    tracker: DialogueStateTracker, all_flows: FlowsList
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

        if command_type == "set slot":
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
    commands: List[Command], tracker: DialogueStateTracker, all_flows: FlowsList
) -> Dict[str, Any]:
    commands_list = []

    for command in commands:
        command_type = command.command()
        command_as_dict = command.as_dict()

        if command_type == "set slot":
            command_as_dict.pop("value", None)

        commands_list.append(command_as_dict)

    return {"commands": str(commands_list), "module_name": "command_processor"}


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

        if command_type == "set slot":
            slot_value = command_as_dict.pop("value", None)
            command_as_dict["is_slot_value_missing_or_none"] = slot_value is None

        commands_list.append(command_as_dict)

    startable_flow_ids = [flow.id for flow in startable_flows.underlying_flows]

    return {
        "commands": json.dumps(commands_list),
        "startable_flow_ids": json.dumps(startable_flow_ids),
    }
