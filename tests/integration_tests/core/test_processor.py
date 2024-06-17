import uuid
from typing import Callable
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pytest import MonkeyPatch, CaptureFixture

from rasa.core.agent import Agent
from rasa.core.channels import UserMessage
from rasa.dialogue_understanding.commands import SetSlotCommand, StartFlowCommand
from rasa.dialogue_understanding.commands.set_slot_command import SetSlotExtractor

from rasa.shared.core.events import SlotSet
from rasa.utils.endpoints import EndpointConfig


@pytest.fixture(scope="session")
@patch("langchain.vectorstores.faiss.FAISS.from_documents")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
async def trained_calm_slot_mappings_bot(
    mock_flow_search_create_embedder: Mock,
    mock_from_documents: Mock,
    trained_async: Callable,
) -> str:
    mock_flow_search_create_embedder.return_value = Mock()
    mock_from_documents.return_value = Mock()
    return await trained_async(
        domain="data/test_calm_slot_mappings/domain.yml",
        config="data/test_calm_slot_mappings/config.yml",
        training_files=[
            "data/test_calm_slot_mappings/data/flows.yml",
            "data/test_calm_slot_mappings/data/nlu.yml",
        ],
    )


@pytest.fixture
@patch("langchain.vectorstores.faiss.FAISS.load_local")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
async def calm_slot_mappings_agent(
    mock_flow_search_create_embedder: Mock,
    mock_load_local: Mock,
    trained_calm_slot_mappings_bot: str,
) -> Agent:
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
        "Yes",
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
                value=True,
                extractor=SetSlotExtractor.NLU.value,
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
    mock_llm_based_router_generate_answer_CALM: AsyncMock,
) -> None:
    """Test that custom slot mappings are validated correctly.

    If the custom action action_ask_<slot_name> is not defined in the domain,
    FlowPolicy will trigger pattern_internal_error.
    """
    monkeypatch.setattr(
        "rasa.dialogue_understanding.coexistence.llm_based_router.LLMBasedRouter._generate_answer_using_llm",
        mock_llm_based_router_generate_answer_CALM,
    )

    sender_id = uuid.uuid4().hex
    processor = calm_slot_mappings_agent.processor
    slot_name = "loyalty_points"

    tracker = await processor.get_tracker(sender_id)
    assert tracker.active_flow is None

    user_msg = "I would like to pay with my loyalty points."
    await processor.handle_message(UserMessage(user_msg, sender_id=sender_id))

    tracker = await processor.get_tracker(sender_id)
    assert tracker.active_flow == "pattern_internal_error"
    assert tracker.get_slot(slot_name) is None
