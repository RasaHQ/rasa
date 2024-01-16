from datetime import datetime
from typing import Any, Sequence, Text, Tuple

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rasa.core.policies.policy import PolicyPrediction
from rasa.engine.graph import ExecutionContext, GraphSchema
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.events import (
    ActionExecuted,
    ActionReverted,
    AgentUttered,
    AllSlotsReset,
    BotUttered,
    ConversationPaused,
    ConversationResumed,
    DialogueStackUpdated,
    FlowCancelled,
    FlowCompleted,
    FlowInterrupted,
    FlowResumed,
    FlowStarted,
    FollowupAction,
    ReminderScheduled,
    Restarted,
    SessionStarted,
    SlotSet,
    StoryExported,
    UserUtteranceReverted,
    UserUttered,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import (
    MockGraphComponent,
    MockGraphNode,
)

training_data_chit_chat = TrainingData(
    training_examples=[
        Message(
            data={
                "text": "hi",
                "message_id": "test_id",
                "metadata": {},
                "commands": [{"command": "chitchat"}],
            }
        ),
    ]
)

training_data_flow = TrainingData(
    training_examples=[
        Message(
            data={
                "text": "send money to Tom",
                "message_id": "test_id",
                "metadata": {},
                "commands": [{"flow": "transfer_money", "command": "start flow"}],
            }
        ),
    ]
)

training_data_slot = TrainingData(
    training_examples=[
        Message(
            data={
                "text": "send money to Tom",
                "message_id": "test_id",
                "metadata": {},
                "commands": [
                    {
                        "name": "transfer_money_recipient",
                        "value": "tom",
                        "command": "set slot",
                    },
                ],
            }
        ),
    ]
)

training_data_flow_and_slot = TrainingData(
    training_examples=[
        Message(
            data={
                "text": "send money to Tom",
                "message_id": "test_id",
                "metadata": {},
                "commands": [
                    {"flow": "transfer_money", "command": "start flow"},
                    {
                        "name": "transfer_money_recipient",
                        "value": "tom",
                        "command": "set slot",
                    },
                ],
            }
        ),
    ]
)

patch = (
    '[{"op": "replace", "path": "/0/step_id", "value": "0_action_trigger_chitchat"}, '
    '{"op": "add", "path": "/0", "value": '
    '{"frame_id": "some-frame-id", "flow_id": "foo", '
    '"step_id": "first_step", "frame_type": "regular", '
    '"utter": "utter-me", "type": "flow"}}]'
)
default_policy_prediction = PolicyPrediction([], "FlowPolicy")
policy_prediction_with_optional_events_dialogue_stack_updated = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[DialogueStackUpdated("someupdate")],
)
policy_prediction_with_optional_events_dialogue_stack_updated2 = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[DialogueStackUpdated(patch)],
)
policy_prediction_with_user_uttered_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[UserUttered("/greet", {"name": "greet", "confidence": 1.0}, [])],
)
policy_prediction_with_bot_uttered_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[BotUttered("my_text", {"my_data": 1})],
)
policy_prediction_with_agent_uttered_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[AgentUttered("my_text", "my_data")],
)
policy_prediction_with_slotset_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[SlotSet("my_slot", "value")],
)
policy_prediction_with_flow_started_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[FlowStarted("my_flow")],
)
policy_prediction_with_flow_interrupted_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[FlowInterrupted("my_flow", "my_step")],
)
policy_prediction_with_flow_resumed_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[FlowResumed("my_flow", "my_step")],
)
policy_prediction_with_flow_completed_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[FlowCompleted("my_flow", "my_step")],
)
policy_prediction_with_flow_cancelled_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[FlowCancelled("my_flow", "my_step")],
)
policy_prediction_with_restarted_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[Restarted()],
)
policy_prediction_with_all_slots_reset_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[AllSlotsReset()],
)
policy_prediction_with_conversation_paused_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[ConversationPaused()],
)
policy_prediction_with_conversation_resumed_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[ConversationResumed()],
)
policy_prediction_with_story_exported_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[StoryExported()],
)
policy_prediction_with_action_reverted_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[ActionReverted()],
)
policy_prediction_with_user_utterance_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[UserUtteranceReverted()],
)
policy_prediction_with_session_started_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[SessionStarted()],
)
policy_prediction_with_action_executed_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[ActionExecuted("my_action")],
)
policy_prediction_with_follow_up_action_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[FollowupAction("my_action")],
)
policy_prediction_with_reminder_scheduled_event = PolicyPrediction(
    [],
    "FlowPolicy",
    optional_events=[ReminderScheduled("my_intent", datetime.now())],
)


def test_tracing(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    instrumentation.instrument(
        tracer_provider,
        graph_node_class=MockGraphNode,
    )

    node_name = "mock_node"
    component_class = MockGraphComponent
    fn_name = "mock_fn"

    expected_attributes = {
        "node_name": node_name,
        "component_class": component_class.__name__,
        "fn_name": fn_name,
    }

    graph_node = MockGraphNode(
        node_name=node_name,
        component_class=component_class,
        constructor_name="create",
        component_config={},
        fn_name=fn_name,
        inputs={},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "1"),
    )

    graph_node()

    captured_spans: Sequence[
        ReadableSpan
    ] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockGraphNode.MockGraphComponent"
    assert captured_span.attributes == expected_attributes


@pytest.mark.parametrize(
    "inputs_from_previous_nodes, expected",
    [
        (
            ("run_LLMCommandGenerator0", training_data_chit_chat.training_examples),
            {
                "commands": "['chitchat']",
            },
        ),
        (
            ("run_LLMCommandGenerator0", training_data_flow.training_examples),
            {
                "commands": "['start flow']",
                "flow_name": "transfer_money",
            },
        ),
        (
            ("run_LLMCommandGenerator0", training_data_slot.training_examples),
            {
                "commands": "['set slot']",
                "slot_name": "transfer_money_recipient",
            },
        ),
        (
            ("run_LLMCommandGenerator0", training_data_flow_and_slot.training_examples),
            {
                "commands": "['start flow', 'set slot']",
                "flow_name": "transfer_money",
                "slot_name": "transfer_money_recipient",
            },
        ),
        (("somethingelse", training_data_chit_chat.training_examples), {}),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                default_policy_prediction,
            ),
            {"policy": "FlowPolicy"},
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_optional_events_dialogue_stack_updated,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'DialogueStackUpdated'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_optional_events_dialogue_stack_updated2,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'DialogueStackUpdated'}",
                "flows": "{'foo'}",
                "utters": "{'utter-me'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_user_uttered_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'UserUttered'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_bot_uttered_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'BotUttered'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_agent_uttered_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'AgentUttered'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_slotset_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'SlotSet'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_flow_started_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'FlowStarted'}",
                "flows": "{'my_flow'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_flow_interrupted_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'FlowInterrupted'}",
                "flows": "{'my_flow'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_flow_resumed_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'FlowResumed'}",
                "flows": "{'my_flow'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_flow_completed_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'FlowCompleted'}",
                "flows": "{'my_flow'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_flow_cancelled_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'FlowCancelled'}",
                "flows": "{'my_flow'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_restarted_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'Restarted'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_all_slots_reset_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'AllSlotsReset'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_conversation_paused_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'ConversationPaused'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_conversation_resumed_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'ConversationResumed'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_story_exported_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'StoryExported'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_action_reverted_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'ActionReverted'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_user_utterance_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'UserUtteranceReverted'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_session_started_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'SessionStarted'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_action_executed_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'ActionExecuted'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_follow_up_action_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'FollowupAction'}",
            },
        ),
        (
            (
                "run_rasa.core.policies.flow_policy.FlowPolicy0",
                policy_prediction_with_reminder_scheduled_event,
            ),
            {
                "policy": "FlowPolicy",
                "optional_events": "{'ReminderScheduled'}",
            },
        ),
    ],
)
def test_tracing_with_inputs_from_previous_nodes(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    inputs_from_previous_nodes: Tuple[Text, Any],
    expected: Text,
) -> None:
    instrumentation.instrument(
        tracer_provider,
        graph_node_class=MockGraphNode,
    )

    node_name = "mock_node"
    component_class = MockGraphComponent
    fn_name = "mock_fn"

    expected_attributes = {
        "node_name": node_name,
        "component_class": component_class.__name__,
        "fn_name": fn_name,
    }

    graph_node = MockGraphNode(
        node_name=node_name,
        component_class=component_class,
        constructor_name="create",
        component_config={},
        fn_name=fn_name,
        inputs={},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "1"),
    )

    graph_node(inputs_from_previous_nodes)

    captured_spans: Sequence[
        ReadableSpan
    ] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockGraphNode.MockGraphComponent"

    expected_attributes.update(expected)  # type: ignore[arg-type]
    assert captured_span.attributes == expected_attributes
