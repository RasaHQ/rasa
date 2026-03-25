import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest
import structlog
from pytest import CaptureFixture, MonkeyPatch

from rasa.core.actions.action import RemoteAction
from rasa.core.agent import Agent
from rasa.core.channels import CollectingOutputChannel, UserMessage
from rasa.core.policies.policy import PolicyPrediction
from rasa.core.processor import MessageProcessor
from rasa.dialogue_understanding.commands import (
    CorrectedSlot,
    CorrectSlotsCommand,
    SetSlotCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.commands.set_slot_command import SetSlotExtractor
from rasa.shared.constants import DEFAULT_SENDER_ID
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_SESSION_START_NAME,
    USER_INTENT_SESSION_START,
)
from rasa.shared.core.domain import SessionConfig
from rasa.shared.core.events import (
    ActionExecuted,
    ConversationInactive,
    ConversationResumed,
    Event,
    SessionEnded,
    SessionStarted,
    SlotSet,
    UserUttered,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import METADATA_SESSION_ID
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.utils.io import read_file
from rasa.utils.endpoints import EndpointConfig
from tests.conftest import TrainedAsync
from tests.utilities import filter_logs, flows_from_str


@pytest.fixture(scope="session")
@patch("langchain_community.vectorstores.faiss.FAISS.save_local")
@patch("langchain_community.vectorstores.faiss.FAISS.from_documents")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_llm_client")
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_embedder")
async def trained_calm_slot_mappings_bot(
    mock_try_instantiate_llm_client: Mock,
    mock_try_instantiate_embedder: Mock,
    mock_save_local: Mock,
    mock_from_documents: Mock,
    mock_flow_search_create_embedder: Mock,
    trained_async: TrainedAsync,
) -> str:
    mock_try_instantiate_llm_client.return_value = Mock()
    mock_flow_search_create_embedder.return_value = Mock()
    mock_from_documents.return_value = Mock()
    mock_save_local.return_value = Mock()
    return await trained_async(
        domain="data/test_calm_slot_mappings/domain.yml",
        config="data/test_calm_slot_mappings/config.yml",
        training_files=[
            "data/test_calm_slot_mappings/data/flows.yml",
            "data/test_calm_slot_mappings/data/nlu.yml",
        ],
    )


@pytest.fixture
@patch("langchain_community.vectorstores.faiss.FAISS.load_local")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_llm_client")
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_embedder")
async def calm_slot_mappings_agent(
    mock_try_instantiate_llm_client: Mock,
    mock_try_instantiate_embedder: Mock,
    mock_flow_search_create_embedder: Mock,
    mock_load_local: AsyncMock,
    trained_calm_slot_mappings_bot: str,
) -> Agent:
    mock_try_instantiate_llm_client.return_value = Mock()
    mock_flow_search_create_embedder.return_value = Mock()
    mock_try_instantiate_embedder.return_value = Mock()
    mock_load_local.return_value = AsyncMock()
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    return Agent.load(
        model_path=trained_calm_slot_mappings_bot, action_endpoint=endpoint
    )


@pytest.fixture
def mock_llm_based_router_generate_answer_CALM() -> AsyncMock:
    return AsyncMock(return_value="A")


@pytest.fixture
def mock_llm_based_router_generate_answer_NLU() -> AsyncMock:
    return AsyncMock(return_value="C")


@pytest.fixture
def mock_filter_flows(*args, **kwargs) -> AsyncMock:
    return AsyncMock(return_value=FlowsList([]))


async def _resume_with_event(processor) -> None:
    """Helper to resume with ConversationResumed event."""
    tracker = await processor.get_tracker(DEFAULT_SENDER_ID)
    tracker.update(ConversationResumed())
    await processor.save_tracker(tracker)


async def _resume_with_user_message(processor) -> None:
    """Helper to resume with UserUttered message."""
    await processor.handle_message(
        UserMessage("resume conversation", sender_id=DEFAULT_SENDER_ID)
    )


def _set_session_config(
    processor: MessageProcessor,
    start_session_after_expiry: bool,
    session_expiration_time: float = 60,
) -> None:
    processor.domain.session_config = SessionConfig(
        session_expiration_time=session_expiration_time,
        carry_over_slots=True,
        start_session_after_expiry=start_session_after_expiry,
    )


async def test_processor_handle_message_calm_slots_with_nlu_pipeline(
    calm_slot_mappings_agent: Agent,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture,
    mock_llm_based_router_generate_answer_CALM: AsyncMock,
):
    """Test the mechanism that the processor uses to fill slots during coexistence.

    The processor should use NLUCommandAdapter to fill slots when the coexistence
    router routes user message to CALM.
    """
    monkeypatch.setattr(
        "rasa.dialogue_understanding.coexistence.llm_based_router.LLMBasedRouter._generate_answer_using_llm",
        mock_llm_based_router_generate_answer_CALM,
    )

    sender_id = uuid.uuid4().hex
    processor = calm_slot_mappings_agent.processor

    user_messages = [
        "I would like to order a diavola pizza",
        "2 please",
        "12 Elm Street",
        "/SetSlots(order_confirmation=True)",
    ]

    response_texts = [
        "How many pizzas would you like to order?",
        "What is the delivery address?",
        "You have put in a order for 2.0 diavola pizzas. "
        "Please confirm these details are correct?",
        "Thank you for your order. Your pizza will be delivered "
        "to 12 Elm Street in 30 minutes.",
    ]

    expected_commands = [
        [
            StartFlowCommand(flow="order_pizza").as_dict(),
            SetSlotCommand(
                name="pizza", value="diavola", extractor=SetSlotExtractor.NLU.value
            ).as_dict(),
        ],
        [
            SetSlotCommand(
                name="num_pizza", value="2", extractor=SetSlotExtractor.NLU.value
            ).as_dict()
        ],
        [
            SetSlotCommand(
                name="address",
                value="12 Elm Street",
                extractor=SetSlotExtractor.NLU.value,
            ).as_dict(),
        ],
        [
            SetSlotCommand(
                name="order_confirmation",
                value="True",
                extractor=SetSlotExtractor.COMMAND_PAYLOAD_READER.value,
            ).as_dict()
        ],
    ]

    for i, user_msg in enumerate(user_messages):
        response = await processor.handle_message(
            UserMessage(user_msg, sender_id=sender_id)
        )

        assert response[0].get("text") == response_texts[i]

        tracker = await processor.get_tracker(sender_id)
        assert tracker.latest_message is not None
        assert tracker.latest_message.text == user_msg
        assert all(
            command in tracker.latest_message.commands
            for command in expected_commands[i]
        )

        captured = capsys.readouterr()
        debug_log = (
            "action_extract_slot=action_extract_slots "
            "len_extraction_events=0 rasa_events=[]"
        )
        assert debug_log in captured.out


async def test_processor_handle_message_calm_slots_coexistence_nlu(
    calm_slot_mappings_agent: Agent,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture,
    mock_llm_based_router_generate_answer_NLU: AsyncMock,
):
    """Test the mechanism that the processor uses to fill slots during coexistence.

    The processor should use the action_extract_slots to fill slots
    when the coexistence router routes user message to NLU.
    """
    monkeypatch.setattr(
        "rasa.dialogue_understanding.coexistence.llm_based_router.LLMBasedRouter._generate_answer_using_llm",
        mock_llm_based_router_generate_answer_NLU,
    )

    sender_id = uuid.uuid4().hex
    processor = calm_slot_mappings_agent.processor
    slot_name = "num_tickets"
    slot_value = "3"

    tracker = await processor.get_tracker(sender_id)
    assert tracker.get_slot(slot_name) is None

    user_msg = f"Do you have {slot_value} tickets left for the store opening?"
    await processor.handle_message(UserMessage(user_msg, sender_id=sender_id))
    captured = capsys.readouterr()
    debug_log = (
        f"action_extract_slot=action_extract_slots len_extraction_events=1 "
        f"rasa_events=[SlotSet(key: {slot_name}, value: {slot_value})]"
    )
    assert debug_log in captured.out

    tracker = await processor.get_tracker(sender_id)
    assert SlotSet(slot_name, slot_value) in tracker.events
    assert tracker.get_slot(slot_name) == slot_value


async def test_processor_handle_message_calm_slots_custom_action_invalid(
    calm_slot_mappings_agent: Agent,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture,
    mock_llm_based_router_generate_answer_CALM: AsyncMock,
    mock_filter_flows: AsyncMock,
    llm_response_object: LLMResponse,
) -> None:
    """Test that controlled slot mappings function correctly."""
    monkeypatch.setattr(
        "rasa.dialogue_understanding.coexistence.llm_based_router.LLMBasedRouter._generate_answer_using_llm",
        mock_llm_based_router_generate_answer_CALM,
    )

    monkeypatch.setattr(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.filter_flows",
        mock_filter_flows,
    )

    async def mock_invoke_llm(*args, **kwargs):
        llm_response_object.choices = ["StartFlow(authenticate_user)"]
        return llm_response_object

    monkeypatch.setattr(
        "rasa.dialogue_understanding.generator.llm_command_generator.SingleStepLLMCommandGenerator.invoke_llm",
        mock_invoke_llm,
    )

    sender_id = uuid.uuid4().hex
    processor = calm_slot_mappings_agent.processor

    tracker = await processor.get_tracker(sender_id)
    assert tracker.active_flow is None

    user_messages = [
        "I would like to login.",
        "/SetSlots(is_member=True)",
    ]
    bot_messages = ["Are you a member?", "You have successfully logged in."]
    for i, msg in enumerate(user_messages):
        responses = await processor.handle_message(
            UserMessage(msg, sender_id=sender_id)
        )
        assert responses[0].get("text") == bot_messages[i]


async def test_processor_handle_message_calm_corrections_for_NLU_slots(
    calm_slot_mappings_agent: Agent,
    monkeypatch: MonkeyPatch,
    mock_llm_based_router_generate_answer_CALM: AsyncMock,
):
    """Test that corrections for NLU slots are correctly handled."""
    monkeypatch.setattr(
        "rasa.dialogue_understanding.coexistence.llm_based_router.LLMBasedRouter._generate_answer_using_llm",
        mock_llm_based_router_generate_answer_CALM,
    )

    sender_id = uuid.uuid4().hex
    processor = calm_slot_mappings_agent.processor

    user_messages = [
        "I would like to order a pepperoni pizza",
        "1 please",
        "Actually can I get a margherita pizza instead?",
    ]

    response_texts = [
        "How many pizzas would you like to order?",
        "What is the delivery address?",
        "Ok, I am updating pizza to margherita respectively.",
    ]

    expected_commands = [
        [
            StartFlowCommand(flow="order_pizza").as_dict(),
            SetSlotCommand(
                name="pizza", value="pepperoni", extractor=SetSlotExtractor.NLU.value
            ).as_dict(),
        ],
        [
            SetSlotCommand(
                name="num_pizza", value="1", extractor=SetSlotExtractor.NLU.value
            ).as_dict()
        ],
        [
            CorrectSlotsCommand(
                [
                    CorrectedSlot(
                        name="pizza",
                        value="margherita",
                        filled_by=SetSlotExtractor.NLU.value,
                    )
                ]
            ).as_dict(),
        ],
    ]

    for i, user_msg in enumerate(user_messages):
        response = await processor.handle_message(
            UserMessage(user_msg, sender_id=sender_id)
        )

        assert response[0].get("text") == response_texts[i]

        tracker = await processor.get_tracker(sender_id)
        assert tracker.latest_message is not None
        assert tracker.latest_message.text == user_msg
        assert all(
            command in tracker.latest_message.commands
            for command in expected_commands[i]
        )
    else:
        # Check that the slot was correctly updated after the latest bot response
        tracker = await processor.get_tracker(sender_id)
        assert tracker.get_slot("pizza") == "margherita"


async def test_processor_handle_message_calm_cannot_handle_command(
    calm_slot_mappings_agent: Agent,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture,
    mock_llm_based_router_generate_answer_CALM: AsyncMock,
    mock_filter_flows: AsyncMock,
    llm_response_object: LLMResponse,
):
    """Test the skipping mechanism for SetSlot commands from LLM command generators.

    The command processor should skip SetSlot commands from LLM command generators
    when the slot has a nlu-based slot mapping and instead issue CannotHandle command.
    """
    monkeypatch.setattr(
        "rasa.dialogue_understanding.coexistence.llm_based_router.LLMBasedRouter._generate_answer_using_llm",
        mock_llm_based_router_generate_answer_CALM,
    )

    monkeypatch.setattr(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.filter_flows",
        mock_filter_flows,
    )

    async def mock_invoke_llm(*args, **kwargs):
        llm_response_object.choices = ["SetSlot(payment_option, credit card)"]
        return llm_response_object

    monkeypatch.setattr(
        "rasa.dialogue_understanding.generator.llm_command_generator.LLMCommandGenerator.invoke_llm",
        mock_invoke_llm,
    )

    sender_id = uuid.uuid4().hex
    processor = calm_slot_mappings_agent.processor

    user_messages = [
        "I would like to pay for my order now.",
        "Could I use credit card?",
    ]

    expected_response = [
        "How would you like to pay for your order?",
        "I’m sorry I am unable to understand you, could you please rephrase?",
    ]

    expected_commands = [
        [
            StartFlowCommand(flow="payment_flow").as_dict(),
        ],
        [
            SetSlotCommand(
                name="payment_option",
                value="credit card",
                extractor=SetSlotExtractor.LLM.value,
            ).as_dict()
        ],
        [],
    ]

    for i, user_message in enumerate(user_messages):
        response = await processor.handle_message(
            UserMessage(user_message, sender_id=sender_id)
        )
        assert response[0].get("text") == expected_response[i]

        tracker = await processor.get_tracker(sender_id)
        assert tracker.latest_message is not None
        assert tracker.latest_message.text == user_message
        assert all(
            command in tracker.latest_message.commands
            for command in expected_commands[i]
        )
        assert tracker.get_slot("payment_option") is None

    captured = capsys.readouterr()
    reason = (
        "A command generator attempted to set a slot with a value extracted "
        "by an extractor that is incompatible with the slot mapping type."
    )
    command_processor_debug_log = f"CannotHandleCommand(reason='{reason}')"
    assert command_processor_debug_log in captured.out


@pytest.fixture(scope="session")
@patch("langchain_community.vectorstores.faiss.FAISS.save_local")
@patch("langchain_community.vectorstores.faiss.FAISS.from_documents")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_llm_client")
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_embedder")
async def trained_force_slot_filling_bot(
    mock_try_instantiate_llm_client: Mock,
    mock_try_instantiate_embedder: Mock,
    mock_save_local: Mock,
    mock_from_documents: Mock,
    mock_flow_search_create_embedder: Mock,
    trained_async: TrainedAsync,
) -> str:
    mock_try_instantiate_llm_client.return_value = Mock()
    mock_try_instantiate_embedder.return_value = Mock()
    mock_flow_search_create_embedder.return_value = Mock()
    mock_from_documents.return_value = Mock()
    mock_save_local.return_value = Mock()
    return await trained_async(
        domain="data/test_force_slot_filling/domain.yml",
        config="data/test_force_slot_filling/config.yml",
        training_files=[
            "data/test_force_slot_filling/data/",
        ],
    )


async def mocked_filter_flows(*args, **kwargs) -> FlowsList:
    return flows_from_str(read_file("data/test_force_slot_filling/data/flows.yml"))


@pytest.fixture
@patch("langchain_community.vectorstores.faiss.FAISS.load_local")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_llm_client")
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_embedder")
async def calm_force_slot_filling_agent(
    mock_try_instantiate_llm_client: Mock,
    mock_try_instantiate_embedder: Mock,
    mock_flow_search_create_embedder: Mock,
    mock_load_local: AsyncMock,
    trained_force_slot_filling_bot: str,
    monkeypatch: MonkeyPatch,
) -> Agent:
    mock_try_instantiate_llm_client.return_value = Mock()
    mock_flow_search_create_embedder.return_value = Mock()
    mock_try_instantiate_embedder.return_value = Mock()
    mock_load_local.return_value = AsyncMock()
    monkeypatch.setattr(
        "rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator.SingleStepLLMCommandGenerator.filter_flows",
        mocked_filter_flows,
    )
    endpoint = EndpointConfig(actions_module="data.test_force_slot_filling.actions")
    return Agent.load(
        model_path=trained_force_slot_filling_bot, action_endpoint=endpoint
    )


async def test_processor_fill_controlled_slot_run_action_every_turn_enabled(
    calm_force_slot_filling_agent: Agent,
) -> None:
    processor = calm_force_slot_filling_agent.processor
    sender_id = uuid.uuid4().hex

    await processor.handle_message(
        UserMessage("I would like to order 1 pepperoni pizza.", sender_id=sender_id)
    )

    tracker = await processor.tracker_store.get_or_create_tracker(sender_id)
    assert tracker.get_slot("action_slot") == 123


async def test_processor_force_slot_filling_from_text(
    calm_force_slot_filling_agent: Agent,
) -> None:
    """Assistant should skip running the NLU graph.

    We test with the slot `address` which has the `force_slot_filling`
    property enabled to True and the `from_text` mapping.
    In this scenario, the NLU graph should not be run and the slot is filled
    directly via the same mechanism as a deterministic button payload.
    """
    processor = calm_force_slot_filling_agent.processor
    sender_id = uuid.uuid4().hex

    user_messages = ["I would like to order 1 pepperoni pizza.", "1 Maple Avenue"]

    response = await processor.handle_message(
        UserMessage(user_messages[0], sender_id=sender_id)
    )
    assert response[0].get("text") == "What is the delivery address?"

    response = await processor.handle_message(
        UserMessage(user_messages[1], sender_id=sender_id)
    )
    assert (
        response[0].get("text") == "You have put in a order for 1.0 pepperoni pizzas. "
        "Please confirm these details are correct?"
    )

    tracker = await processor.get_tracker(sender_id)
    assert tracker.get_slot("address") == "1 Maple Avenue"
    assert tracker.latest_message.commands == [
        SetSlotCommand(
            name="address",
            value="1 Maple Avenue",
            extractor=SetSlotExtractor.COMMAND_PAYLOAD_READER.value,
        ).as_dict(),
    ]


async def test_processor_force_slot_filling_non_from_text(
    calm_force_slot_filling_agent: Agent,
) -> None:
    """Assistant should only fill slots and not process other commands.

    We test with the slot `order_confirmation` which has the `force_slot_filling`
    property enabled to True and a different slot mapping to `from_text` mapping.
    In this scenario, any command other than `SetSlot` is filtered out
    by the Command Generator.
    """
    processor = calm_force_slot_filling_agent.processor
    sender_id = uuid.uuid4().hex

    user_messages = [
        "I would like to order 1 pepperoni pizza.",
        "31 Blueberry Lane",
        "Nevermind, cancel pizza order.",
    ]

    response = await processor.handle_message(
        UserMessage(user_messages[0], sender_id=sender_id)
    )
    assert response[0].get("text") == "What is the delivery address?"

    response = await processor.handle_message(
        UserMessage(user_messages[1], sender_id=sender_id)
    )
    assert (
        response[0].get("text") == "You have put in a order for 1.0 pepperoni pizzas. "
        "Please confirm these details are correct?"
    )

    response = await processor.handle_message(
        UserMessage(user_messages[2], sender_id=sender_id)
    )
    assert response[0].get("text") == "Your order has been cancelled."

    tracker = await processor.get_tracker(sender_id)
    assert tracker.latest_message.commands == [
        SetSlotCommand(
            name="order_confirmation", value=False, extractor=SetSlotExtractor.NLU.value
        ).as_dict(),
    ]


@pytest.mark.parametrize(
    "start_session_after_expiry,expected_session_started_count",
    [(True, 2), (False, 1)],
)
async def test_processor_handles_expired_session_depends_on_session_config(
    default_agent: Agent,
    monkeypatch: MonkeyPatch,
    start_session_after_expiry: bool,
    expected_session_started_count: int,
):
    """Test expired session: with start_session_after_expiry True a new session
    is started; with False the same session continues."""
    processor = default_agent.processor

    # Configure session expiration (1 minute) and whether to start new session on expiry
    _set_session_config(processor, start_session_after_expiry, 1)

    await processor.handle_message(UserMessage("hello", sender_id=DEFAULT_SENDER_ID))

    tracker = await processor.get_tracker(DEFAULT_SENDER_ID)
    initial_events_count = len(tracker.events)
    assert SessionStarted() in tracker.events

    # Simulate session expiration
    monkeypatch.setattr(processor, "_has_session_expired", lambda _: True)
    tracker.update(ConversationInactive())
    await processor.save_tracker(tracker)

    await _resume_with_user_message(processor)

    tracker = await processor.tracker_store.retrieve_full_tracker(DEFAULT_SENDER_ID)
    assert tracker is not None
    assert len(tracker.events) > initial_events_count

    session_started_count = sum(
        1 for event in tracker.events if isinstance(event, SessionStarted)
    )
    assert session_started_count == expected_session_started_count


@pytest.mark.parametrize("start_session_after_expiry", [True, False])
async def test_processor_non_expired_session_continues_same_session(
    default_agent: Agent,
    start_session_after_expiry: bool,
):
    """When session has not expired, no new SessionStarted is added (same session
    continues) regardless of start_session_after_expiry."""
    processor = default_agent.processor

    _set_session_config(processor, start_session_after_expiry)

    await processor.handle_message(UserMessage("hello", sender_id=DEFAULT_SENDER_ID))
    await _resume_with_user_message(processor)

    tracker = await processor.tracker_store.retrieve_full_tracker(DEFAULT_SENDER_ID)
    assert tracker is not None

    session_started_count = sum(
        1 for event in tracker.events if isinstance(event, SessionStarted)
    )
    assert session_started_count == 1


async def test_action_session_start_executes_once_on_session_start_intent(
    default_agent: Agent,
):
    """Test that action_session_start runs only once when /session_start is sent.

    When a user sends /session_start to start a new session, both
    MessageProcessor and FlowPolicy would previously trigger
    action_session_start. This test verifies it runs exactly once.
    """
    processor = default_agent.processor
    sender_id = uuid.uuid4().hex

    await processor.handle_message(
        UserMessage(f"/{USER_INTENT_SESSION_START}", sender_id=sender_id)
    )

    tracker = await processor.tracker_store.retrieve_full_tracker(sender_id)
    assert tracker is not None

    session_started_count = sum(
        1 for event in tracker.events if isinstance(event, SessionStarted)
    )
    assert session_started_count == 1


@pytest.mark.parametrize(
    "setup_resume", [_resume_with_user_message, _resume_with_event]
)
async def test_processor_handles_inactive_session_resumes(
    default_agent: Agent,
    setup_resume,
):
    """Test that inactive sessions can be resumed with certain events."""
    processor = default_agent.processor

    await processor.handle_message(UserMessage("hello", sender_id=DEFAULT_SENDER_ID))

    tracker = await processor.get_tracker(DEFAULT_SENDER_ID)
    tracker.update(ConversationInactive())
    await processor.save_tracker(tracker)

    tracker = await processor.get_tracker(DEFAULT_SENDER_ID)
    assert tracker.inactive
    assert not tracker.terminated

    await setup_resume(processor)

    tracker = await processor.tracker_store.retrieve_full_tracker(DEFAULT_SENDER_ID)
    assert tracker is not None
    assert not tracker.inactive
    assert not tracker.terminated

    # Verify conversation can continue
    await _resume_with_user_message(processor)
    tracker = await processor.tracker_store.retrieve_full_tracker(DEFAULT_SENDER_ID)
    assert tracker is not None

    user_messages = [e for e in tracker.events if isinstance(e, UserUttered)]
    assert len(user_messages) > 0


async def test_processor_handles_terminated_session_no_restart(
    default_agent: Agent,
):
    """Test that terminated sessions cannot accept new messages."""
    processor = default_agent.processor
    await processor.handle_message(UserMessage("hello", sender_id=DEFAULT_SENDER_ID))

    tracker = await processor.get_tracker(DEFAULT_SENDER_ID)
    tracker.update(SessionEnded())
    await processor.save_tracker(tracker)

    tracker = await processor.get_tracker(DEFAULT_SENDER_ID)
    assert tracker.terminated
    assert not tracker.inactive

    initial_event_count = len(tracker.events)
    initial_session_started_count = sum(
        1 for event in tracker.events if isinstance(event, SessionStarted)
    )

    with structlog.testing.capture_logs() as caplog:
        await _resume_with_user_message(processor)

        logs = filter_logs(
            caplog,
            event="rasa.core.processor.handle_message_with_tracker.terminated_conversation",
            log_level="debug",
        )
        assert len(logs) == 1
        assert "Ignoring message from user as conversation" in logs[0]["event_info"]
        assert "was terminated with a SessionEnded event" in logs[0]["event_info"]

    # Verify no new session was started and state unchanged
    tracker = await processor.tracker_store.retrieve_full_tracker(DEFAULT_SENDER_ID)
    assert tracker is not None
    assert tracker.terminated
    assert len(tracker.events) == initial_event_count

    final_session_started_count = sum(
        1 for event in tracker.events if isinstance(event, SessionStarted)
    )
    assert final_session_started_count == initial_session_started_count


async def test_processor_new_conversation_after_termination(
    default_agent: Agent,
):
    """Test creating a completely new conversation after termination."""
    processor = default_agent.processor

    await processor.handle_message(UserMessage("hello", sender_id=DEFAULT_SENDER_ID))
    tracker_1 = await processor.tracker_store.retrieve_full_tracker(DEFAULT_SENDER_ID)
    assert tracker_1 is not None
    tracker_1.update(SessionEnded())
    await processor.save_tracker(tracker_1)

    tracker_1 = await processor.tracker_store.retrieve_full_tracker(DEFAULT_SENDER_ID)
    assert tracker_1 is not None
    assert tracker_1.terminated

    # Create new conversation with different sender_id
    sender_id = uuid.uuid4().hex
    await processor.handle_message(UserMessage("new conversation", sender_id=sender_id))

    tracker_2 = await processor.tracker_store.retrieve_full_tracker(sender_id)
    assert tracker_2 is not None
    assert not tracker_2.terminated
    assert not tracker_2.inactive
    assert SessionStarted() in tracker_2.events

    user_messages = [e for e in tracker_2.events if isinstance(e, UserUttered)]
    assert len(user_messages) > 0
    assert user_messages[-1].text == "new conversation"

    # Verify first conversation is still terminated
    tracker_1 = await processor.tracker_store.retrieve_full_tracker(DEFAULT_SENDER_ID)
    assert tracker_1 is not None
    assert tracker_1.terminated


@pytest.mark.parametrize(
    "start_session_after_expiry,expected_session_started_count",
    [(True, 2), (False, 1)],
)
async def test_processor_session_expiration_with_inactive_state(
    default_agent: Agent,
    monkeypatch: MonkeyPatch,
    start_session_after_expiry: bool,
    expected_session_started_count: int,
):
    """Test edge case: session 'expires' while in inactive state; with auto_start True
    a new session is started on resume, with False the same session continues."""
    processor = default_agent.processor

    _set_session_config(processor, start_session_after_expiry, 1)

    await processor.handle_message(UserMessage("hello", sender_id=DEFAULT_SENDER_ID))

    tracker = await processor.get_tracker(DEFAULT_SENDER_ID)
    tracker.update(ConversationInactive())
    await processor.save_tracker(tracker)

    tracker = await processor.get_tracker(DEFAULT_SENDER_ID)
    assert tracker.inactive

    monkeypatch.setattr(processor, "_has_session_expired", lambda _: True)

    await _resume_with_user_message(processor)

    tracker = await processor.tracker_store.retrieve_full_tracker(DEFAULT_SENDER_ID)
    assert tracker is not None
    assert not tracker.inactive
    assert not tracker.terminated

    session_started_events = [
        event for event in tracker.events if isinstance(event, SessionStarted)
    ]
    assert len(session_started_events) == expected_session_started_count


async def test_processor_preserves_inactive_state_across_operations(
    default_agent: Agent,
):
    """Test that inactive state persists through processor operations."""
    processor = default_agent.processor

    await processor.handle_message(UserMessage("hello", sender_id=DEFAULT_SENDER_ID))

    tracker = await processor.get_tracker(DEFAULT_SENDER_ID)
    tracker.update(ConversationInactive())
    await processor.save_tracker(tracker)

    tracker = await processor.get_tracker(DEFAULT_SENDER_ID)
    assert tracker.inactive
    assert not tracker.terminated

    # Save and retrieve again
    await processor.save_tracker(tracker)
    tracker = await processor.tracker_store.retrieve_full_tracker(DEFAULT_SENDER_ID)
    assert tracker is not None
    assert tracker.inactive
    assert not tracker.terminated


@pytest.mark.parametrize(
    "event",
    [
        SlotSet("slot", "value"),
        ConversationInactive(),
        ConversationResumed(),
        UserUttered("test"),
    ],
)
async def test_processor_terminated_prevents_tracker_modifications(
    default_agent: Agent,
    event: Event,
):
    """Test that terminated state prevents all tracker modifications."""
    processor = default_agent.processor

    await processor.handle_message(UserMessage("hello", sender_id=DEFAULT_SENDER_ID))

    tracker = await processor.get_tracker(DEFAULT_SENDER_ID)
    tracker.update(SessionEnded())
    await processor.save_tracker(tracker)

    tracker = await processor.get_tracker(DEFAULT_SENDER_ID)
    assert tracker.terminated
    initial_event_count = len(tracker.events)

    with structlog.testing.capture_logs() as caplog:
        tracker.update(event)

        logs = filter_logs(
            caplog,
            event="rasa.shared.core.trackers.dialogue_state_tracker.update_terminated_conversation",
            log_level="warning",
        )
        assert len(logs) == 1
        assert (
            "Ignoring event on a terminated conversation "
            "The conversation was terminated with a "
            "SessionEnded event and cannot be modified." in logs[0]["event_info"]
        )

    assert len(tracker.events) == initial_event_count
    assert tracker.terminated


@pytest.mark.parametrize(
    "setup_resume", [_resume_with_user_message, _resume_with_event]
)
async def test_processor_inactive_state_with_other_events(
    default_agent: Agent, setup_resume
):
    """Test that inactive state persists when non-user events are added."""
    processor = default_agent.processor

    await processor.handle_message(UserMessage("hello", sender_id=DEFAULT_SENDER_ID))

    tracker = await processor.get_tracker(DEFAULT_SENDER_ID)
    tracker.update(ConversationInactive())
    await processor.save_tracker(tracker)

    tracker = await processor.get_tracker(DEFAULT_SENDER_ID)
    assert tracker.inactive

    # Add non-user events, inactive should persist
    tracker.update(SlotSet("some_slot", "value"))
    await processor.save_tracker(tracker)

    tracker = await processor.get_tracker(DEFAULT_SENDER_ID)
    assert tracker.inactive

    # Only UserUttered or ConversationResumed should clear inactive
    await setup_resume(processor)
    tracker = await processor.tracker_store.retrieve_full_tracker(DEFAULT_SENDER_ID)
    assert tracker is not None
    assert not tracker.inactive


async def test_session_id_preserved_across_tracker_store_roundtrip(
    default_agent: Agent,
    monkeypatch: MonkeyPatch,
):
    """Session IDs are preserved when saving and retrieving tracker."""
    channel = CollectingOutputChannel()
    processor = default_agent.processor
    sender_id = uuid.uuid4().hex

    await processor.handle_message(UserMessage("/greet", channel, sender_id))
    tracker = await processor.tracker_store.retrieve(sender_id)
    first_session_id = tracker.current_session_id
    assert first_session_id is not None
    for event in tracker.events:
        assert event.metadata.get(METADATA_SESSION_ID) == first_session_id

    # simulate session expiry
    monkeypatch.setattr(processor, "_has_session_expired", lambda _: True)
    tracker.update(ConversationInactive())
    await processor.save_tracker(tracker)

    await processor.handle_message(UserMessage("/greet", channel, sender_id))

    tracker = await processor.tracker_store.retrieve_full_tracker(sender_id)
    assert tracker is not None
    second_session_id = tracker.current_session_id
    assert second_session_id is not None
    assert second_session_id != first_session_id

    session_started_events = [
        e for e in tracker.events if isinstance(e, SessionStarted)
    ]
    assert len(session_started_events) >= 2
    assert (
        session_started_events[0].metadata.get(METADATA_SESSION_ID) == first_session_id
    )
    assert (
        session_started_events[1].metadata.get(METADATA_SESSION_ID) == second_session_id
    )


async def test_session_id_replay_preserves_original_metadata(
    default_agent: Agent,
    monkeypatch: MonkeyPatch,
):
    """Store roundtrip preserves session_id on events."""
    channel = CollectingOutputChannel()
    processor = default_agent.processor
    sender_id = uuid.uuid4().hex

    await processor.handle_message(UserMessage("/greet", channel, sender_id))
    # simulate session expiry
    monkeypatch.setattr(processor, "_has_session_expired", lambda _: True)
    tracker = await processor.tracker_store.retrieve_full_tracker(sender_id)
    tracker.update(ConversationInactive())
    await processor.save_tracker(tracker)

    await processor.handle_message(UserMessage("/greet", channel, sender_id))

    tracker = await processor.tracker_store.retrieve_full_tracker(sender_id)
    assert tracker is not None
    events = list(tracker.events)
    session_ids_in_events = {
        e.metadata.get(METADATA_SESSION_ID)
        for e in events
        if e.metadata.get(METADATA_SESSION_ID)
    }
    assert len(session_ids_in_events) >= 2, "expected at least two distinct sessions"
    assert tracker.current_session_id is not None
    assert tracker.current_session_id in session_ids_in_events
    events_with_current = [
        e
        for e in events
        if e.metadata.get(METADATA_SESSION_ID) == tracker.current_session_id
    ]
    assert len(events_with_current) >= 1


async def test_session_id_replay_does_not_inject_into_old_events(
    default_agent: Agent,
):
    """Store roundtrip does not inject session_id into events that did not have it."""
    processor = default_agent.processor
    sender_id = uuid.uuid4().hex
    new_session_id = "new-session-id-789"
    stored_events = [
        ActionExecuted(ACTION_SESSION_START_NAME),
        SessionStarted(),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("old message"),
        ActionExecuted(
            ACTION_SESSION_START_NAME, metadata={METADATA_SESSION_ID: new_session_id}
        ),
        SessionStarted(metadata={METADATA_SESSION_ID: new_session_id}),
        ActionExecuted(
            ACTION_LISTEN_NAME, metadata={METADATA_SESSION_ID: new_session_id}
        ),
    ]
    tracker = DialogueStateTracker.from_events(
        sender_id,
        stored_events,
        domain=processor.domain,
    )
    await processor.tracker_store.save(tracker)

    retrieved = await processor.tracker_store.retrieve_full_tracker(sender_id)
    assert retrieved is not None
    events = list(retrieved.events)
    assert len(events) >= 7
    for i in range(4):
        assert METADATA_SESSION_ID not in events[i].metadata
    for i in range(4, 7):
        assert events[i].metadata[METADATA_SESSION_ID] == new_session_id
    assert retrieved.current_session_id == new_session_id


async def test_custom_action_returning_session_ended_cancels_timer(
    default_agent: Agent,
):
    """Custom action returning SessionEnded cancels the inactivity timer.

    Verifies the full pipeline:
    _run_action → execute_side_effects → _handle_session_timer_events → cancel_timer.
    """
    processor = default_agent.processor
    channel = CollectingOutputChannel()

    action_endpoint = EndpointConfig(
        actions_module=(
            "tests_deployment.integration_tests_custom_action_server"
            ".simple_calm_bot.actions.action_end_session"
        )
    )
    remote_action = RemoteAction("action_end_session", action_endpoint)

    tracker = await processor.tracker_store.get_or_create_tracker(DEFAULT_SENDER_ID)
    await processor.timer_manager.schedule_timer(
        sender_id=DEFAULT_SENDER_ID,
        session_id=tracker.current_session_id,
        timeout_seconds=300.0,
        callback=processor.handle_session_timeout,
    )
    assert await processor.timer_manager.get_timer(DEFAULT_SENDER_ID) is not None

    prediction = PolicyPrediction(probabilities=[], policy_name="some_policy")
    await processor._run_action(
        action=remote_action,
        tracker=tracker,
        output_channel=channel,
        nlg=processor.nlg,
        prediction=prediction,
    )

    assert tracker.terminated is True
    assert await processor.timer_manager.get_timer(DEFAULT_SENDER_ID) is None
