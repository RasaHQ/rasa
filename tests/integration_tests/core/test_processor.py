import uuid
from typing import Callable
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pytest import MonkeyPatch, CaptureFixture

from rasa.core.agent import Agent
from rasa.core.channels import UserMessage
from rasa.dialogue_understanding.commands import (
    CorrectSlotsCommand,
    CorrectedSlot,
    SetSlotCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.commands.set_slot_command import SetSlotExtractor
from rasa.dialogue_understanding.processor.command_processor import CANNOT_HANDLE_REASON
from rasa.shared.core.events import BotUttered, SlotSet
from rasa.shared.core.flows import FlowsList
from rasa.utils.endpoints import EndpointConfig


@pytest.fixture(scope="session")
@patch("langchain_community.vectorstores.faiss.FAISS.save_local")
@patch("langchain_community.vectorstores.faiss.FAISS.from_documents")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
@patch(
    "rasa.dialogue_understanding.coexistence.llm_based_router.try_instantiate_llm_client"
)
@patch(
    "rasa.dialogue_understanding.generator.llm_based_command_generator.try_instantiate_llm_client"
)
async def trained_calm_slot_mappings_bot(
    mock_try_instantiate_llm_command_generator_client: Mock,
    mock_try_instantiate_llm_based_router_client: Mock,
    mock_save_local: Mock,
    mock_from_documents: Mock,
    mock_flow_search_create_embedder: Mock,
    trained_async: Callable,
) -> str:
    mock_try_instantiate_llm_command_generator_client.return_value = Mock()
    mock_try_instantiate_llm_based_router_client.return_value = Mock()
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
@patch(
    "rasa.dialogue_understanding.coexistence.llm_based_router.try_instantiate_llm_client"
)
@patch(
    "rasa.dialogue_understanding.generator.llm_based_command_generator.try_instantiate_llm_client"
)
async def calm_slot_mappings_agent(
    mock_try_instantiate_llm_command_generator_client: Mock,
    mock_try_instantiate_llm_based_router_client: Mock,
    mock_flow_search_create_embedder: Mock,
    mock_load_local: Mock,
    trained_calm_slot_mappings_bot: str,
) -> Agent:
    mock_try_instantiate_llm_command_generator_client.return_value = Mock()
    mock_try_instantiate_llm_based_router_client.return_value = Mock()
    mock_flow_search_create_embedder.return_value = Mock()
    mock_load_local.return_value = Mock()
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
        debug_log = "action_extract_slot=action_extract_slots"
        assert debug_log not in captured.out


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
) -> None:
    """Test that custom slot mappings are validated correctly.

    If the custom action action_ask_<slot_name> is not defined in the domain,
    FlowPolicy will first cancel the user flow in progress and then
    trigger pattern_internal_error.
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
        return "StartFlow(authenticate_user)"

    monkeypatch.setattr(
        "rasa.dialogue_understanding.generator.llm_command_generator.LLMCommandGenerator.invoke_llm",
        mock_invoke_llm,
    )

    sender_id = uuid.uuid4().hex
    processor = calm_slot_mappings_agent.processor
    slot_name = "is_member"

    tracker = await processor.get_tracker(sender_id)
    assert tracker.active_flow is None

    user_msg = "I would like to login."
    responses = await processor.handle_message(
        UserMessage(user_msg, sender_id=sender_id)
    )
    assert (
        responses[0].get("text")
        == "Sorry, I am having trouble with that. Please try again in a few minutes."
    )

    tracker = await processor.get_tracker(sender_id)
    assert tracker.get_slot(slot_name) is None
    assert (
        tracker.get_last_event_for(BotUttered).metadata.get("utter_action")
        == "utter_can_do_something_else"
    )

    captured = capsys.readouterr()
    debug_log = "flow.step.run.collect_action_not_found_for_custom_slot_mapping"
    assert debug_log in captured.out


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
                [CorrectedSlot(name="pizza", value="margherita")]
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
        return "SetSlot(payment_option, credit card)"

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
    command_processor_debug_log = (
        f"CannotHandleCommand(reason='{CANNOT_HANDLE_REASON}')"
    )
    assert command_processor_debug_log in captured.out
