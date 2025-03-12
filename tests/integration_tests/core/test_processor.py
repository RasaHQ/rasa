import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pytest import CaptureFixture, MonkeyPatch

from rasa.core.agent import Agent
from rasa.core.channels import UserMessage
from rasa.dialogue_understanding.commands import (
    CorrectedSlot,
    CorrectSlotsCommand,
    SetSlotCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.commands.set_slot_command import SetSlotExtractor
from rasa.dialogue_understanding.processor.command_processor import CANNOT_HANDLE_REASON
from rasa.shared.core.events import BotUttered, SlotSet
from rasa.shared.core.flows import FlowsList
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.utils.io import read_file
from rasa.utils.endpoints import EndpointConfig
from tests.conftest import TrainedAsync
from tests.utilities import flows_from_str


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
    command_processor_debug_log = (
        f"CannotHandleCommand(reason='{CANNOT_HANDLE_REASON}')"
    )
    assert command_processor_debug_log in captured.out


@pytest.fixture(scope="session")
@patch("langchain_community.vectorstores.faiss.FAISS.save_local")
@patch("langchain_community.vectorstores.faiss.FAISS.from_documents")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_llm_client")
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_embedder")
async def trained_handle_digressions_bot(
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
        domain="data/test_handle_digressions/domain.yml",
        config="data/test_handle_digressions/config.yml",
        training_files=[
            "data/test_handle_digressions/data/flows.yml",
        ],
    )


async def mocked_filter_flows(*args, **kwargs) -> FlowsList:
    return flows_from_str(read_file("data/test_handle_digressions/data/flows.yml"))


@pytest.fixture
@patch("langchain_community.vectorstores.faiss.FAISS.load_local")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_llm_client")
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_embedder")
async def calm_handle_digressions_agent(
    mock_try_instantiate_llm_client: Mock,
    mock_try_instantiate_embedder: Mock,
    mock_flow_search_create_embedder: Mock,
    mock_load_local: AsyncMock,
    trained_handle_digressions_bot: str,
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
    endpoint = EndpointConfig(actions_module="data.test_handle_digressions.actions")
    return Agent.load(
        model_path=trained_handle_digressions_bot, action_endpoint=endpoint
    )


async def test_processor_handle_digressions_confirm_digression(
    calm_handle_digressions_agent: Agent,
    monkeypatch: MonkeyPatch,
):
    """Test the mechanism that the processor uses to handle digressions.

    The scenario is as follows:
    1. User starts the order_pizza flow.
    2. User digresses to check account balance.
    3. Bot asks the user if to continue with the order_pizza original flow.
    4. User decides to continue the order_pizza flow.
    """
    sender_id = uuid.uuid4().hex
    processor = calm_handle_digressions_agent.processor

    user_messages = [
        "I would like to order 1 pepperoni pizza.",
        "Before ordering can i check my account balance first?",
        "/SetSlots(continue_previous_flow=True)",
    ]

    flows = ["order_pizza", "pattern_handle_digressions", "order_pizza"]

    for i, user_msg in enumerate(user_messages):
        await processor.handle_message(UserMessage(user_msg, sender_id=sender_id))

        tracker = await processor.get_tracker(sender_id)
        assert tracker.latest_message is not None
        assert tracker.latest_message.text == user_msg
        assert tracker.active_flow == flows[i]
    else:
        actual_responses = []
        for event in tracker.events:
            if isinstance(event, BotUttered):
                actual_responses.append(event.metadata.get("utter_action"))

        assert actual_responses == [
            "utter_ask_address",
            "utter_ask_continue_previous_flow",
            "utter_block_digressions",
            "utter_ask_address",
        ]


async def test_processor_handle_digressions_continue_interruption(
    calm_handle_digressions_agent: Agent,
    monkeypatch: MonkeyPatch,
):
    """Test the mechanism that the processor uses to handle digressions.

    The scenario is as follows:
    1. User starts the order_pizza flow.
    2. User digresses to check account balance.
    3. Bot asks the user if to continue with the order_pizza original flow.
    4. User decides to continue with the digression.
    """
    sender_id = uuid.uuid4().hex
    processor = calm_handle_digressions_agent.processor

    user_messages = [
        "I would like to order 1 pepperoni pizza.",
        "Before ordering can i check my account balance first?",
        "/SetSlots(continue_previous_flow=False)",
    ]

    flows = ["order_pizza", "pattern_handle_digressions", "order_pizza"]

    for i, user_msg in enumerate(user_messages):
        await processor.handle_message(UserMessage(user_msg, sender_id=sender_id))

        tracker = await processor.get_tracker(sender_id)
        assert tracker.latest_message is not None
        assert tracker.latest_message.text == user_msg
        assert tracker.active_flow == flows[i]
    else:
        actual_responses = []
        for event in tracker.events:
            if isinstance(event, BotUttered):
                actual_responses.append(event.metadata.get("utter_action"))

        assert actual_responses == [
            "utter_ask_address",
            "utter_ask_continue_previous_flow",
            "utter_continue_interruption",
            "utter_check_balance",
            "utter_flow_continue_interrupted",
            "utter_ask_address",
        ]


async def test_processor_handle_digressions_block_digression(
    calm_handle_digressions_agent: Agent,
    monkeypatch: MonkeyPatch,
):
    """Test the mechanism that the processor uses to handle digressions.

    The scenario is as follows:
    1. User starts the order_pizza flow.
    2. User digresses to check account balance.
    3. Bot informs the user that they will continue with the pizza order and
    then return to the digression.
    """
    sender_id = uuid.uuid4().hex
    processor = calm_handle_digressions_agent.processor

    user_messages = [
        "I would like to order 1 pepperoni pizza.",
        "1 Clerkenwell Road",
        "Before ordering can i check my account balance first?",
        "/SetSlots(order_confirmation=True)",
    ]

    flows = ["order_pizza", "order_pizza", "order_pizza", None]

    for i, user_msg in enumerate(user_messages):
        await processor.handle_message(UserMessage(user_msg, sender_id=sender_id))

        tracker = await processor.get_tracker(sender_id)
        assert tracker.latest_message is not None
        assert tracker.latest_message.text == user_msg
        assert tracker.active_flow == flows[i]
    else:
        actual_responses = []
        for event in tracker.events:
            if isinstance(event, BotUttered):
                actual_responses.append(event.metadata.get("utter_action"))

        assert actual_responses == [
            "utter_ask_address",
            "utter_ask_order_confirmation",
            "utter_block_digressions",
            "utter_ask_order_confirmation",
            "utter_place_order",
            "utter_flow_continue_interrupted",
            "utter_check_balance",
            "utter_can_do_something_else",
        ]


async def test_processor_handle_digressions_digression_is_not_blocked(
    calm_handle_digressions_agent: Agent,
    monkeypatch: MonkeyPatch,
):
    """Test the mechanism that the processor uses to handle digressions.

    The scenario is as follows:
    1. User starts the payment flow.
    2. User digresses to check account balance.
    3. Bot interrupts the payment flow because check account balance is not listed in
    block_digressions of the card number collect step.
    """
    sender_id = uuid.uuid4().hex
    processor = calm_handle_digressions_agent.processor

    user_messages = [
        "I would like to pay for my electricity bill",
        "/SetSlots(payment_option=card)",
        "Before ordering can i check my account balance first?",
    ]

    for i, user_msg in enumerate(user_messages):
        await processor.handle_message(UserMessage(user_msg, sender_id=sender_id))
    else:
        tracker = await processor.get_tracker(sender_id)
        actual_responses = []
        for event in tracker.events:
            if isinstance(event, BotUttered):
                actual_responses.append(event.metadata.get("utter_action"))

        assert actual_responses == [
            "utter_ask_payment_option",
            "utter_ask_card_number",
            "utter_continue_interruption",
            "utter_check_balance",
            "utter_flow_continue_interrupted",
            "utter_ask_card_number",
        ]


async def test_processor_handle_digressions_trigger_clarification(
    calm_handle_digressions_agent: Agent,
    monkeypatch: MonkeyPatch,
):
    """Test the mechanism that the processor uses to handle digressions.

    The scenario is as follows:
    1. User starts the order_pizza flow.
    2. User digresses the first time.
    3. Bot informs the user that they will continue with the pizza order and
    then return to the digression.
    4. User digresses a second time and then chooses to continue with the pizza order.
    5. Order pizza flow is completed and the bot triggers pattern clarification for the
    user to choose from the pending 2 digressions.
    6. User chooses to check account balance.
    7. Bot triggers the check account balance flow, completes it and then proceeds
    with the payment flow.
    """
    sender_id = uuid.uuid4().hex
    processor = calm_handle_digressions_agent.processor

    user_messages = [
        "I would like to order 1 pepperoni pizza.",
        "Oh can i pay my council tax bill first?",
        "/SetSlots(continue_previous_flow=True)",
        "1 Clerkenwell Road",
        "Before ordering can i check my account balance first?",
        "/SetSlots(order_confirmation=True)",
        "check account balance",
    ]

    for i, user_msg in enumerate(user_messages):
        await processor.handle_message(UserMessage(user_msg, sender_id=sender_id))
    else:
        actual_responses = []
        tracker = await processor.get_tracker(sender_id)
        for event in tracker.events:
            if isinstance(event, BotUttered):
                actual_responses.append(event.metadata.get("utter_action"))

        assert actual_responses == [
            "utter_ask_address",
            "utter_ask_continue_previous_flow",
            "utter_block_digressions",
            "utter_ask_address",
            "utter_ask_order_confirmation",
            "utter_block_digressions",
            "utter_ask_order_confirmation",
            "utter_place_order",
            "utter_clarification_options_rasa",
            "utter_check_balance",
            "utter_flow_continue_interrupted",
            "utter_ask_payment_option",
        ]


async def test_processor_handle_digressions_cancel_clarification_options(
    calm_handle_digressions_agent: Agent,
    monkeypatch: MonkeyPatch,
):
    """Test the mechanism that the processor uses to handle digressions.

    The scenario is as follows:
    1. User starts the order_pizza flow.
    2. User digresses the first time.
    3. Bot informs the user that they will continue with the pizza order and
    then return to the digression.
    4. User digresses a second time and then chooses to continue with the pizza order.
    5. Order pizza flow is completed and the bot triggers pattern clarification for the
    user to choose from the pending 2 digressions.
    6. User chooses to cancel all options.
    7. Bot cancels all pending digressions.
    """
    sender_id = uuid.uuid4().hex
    processor = calm_handle_digressions_agent.processor

    user_messages = [
        "I would like to order 1 pepperoni pizza.",
        "Oh can i pay my council tax bill first?",
        "/SetSlots(continue_previous_flow=True)",
        "1 Clerkenwell Road",
        "Before ordering can i check my account balance first?",
        "/SetSlots(order_confirmation=True)",
        "Cancel all.",
    ]

    for i, user_msg in enumerate(user_messages):
        await processor.handle_message(UserMessage(user_msg, sender_id=sender_id))
    else:
        actual_responses = []
        tracker = await processor.get_tracker(sender_id)
        for event in tracker.events:
            if isinstance(event, BotUttered):
                actual_responses.append(event.metadata.get("utter_action"))

        assert actual_responses == [
            "utter_ask_address",
            "utter_ask_continue_previous_flow",
            "utter_block_digressions",
            "utter_ask_address",
            "utter_ask_order_confirmation",
            "utter_block_digressions",
            "utter_ask_order_confirmation",
            "utter_place_order",
            "utter_clarification_options_rasa",
            "utter_flow_cancelled_rasa",
            "utter_flow_cancelled_rasa",
            "utter_can_do_something_else",
        ]


async def test_processor_fill_controlled_slot_run_action_every_turn_enabled(
    calm_handle_digressions_agent: Agent,
) -> None:
    processor = calm_handle_digressions_agent.processor
    sender_id = uuid.uuid4().hex

    await processor.handle_message(
        UserMessage("I would like to order 1 pepperoni pizza.", sender_id=sender_id)
    )

    tracker = await processor.tracker_store.get_or_create_tracker(sender_id)
    assert tracker.get_slot("action_slot") == 123


async def test_processor_handle_multiple_digressions_continue_with_previous_flow_when_asked(  # noqa: E501
    calm_handle_digressions_agent: Agent,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test how the processor handles multiple duplicate digressions.

    The scenario is as follows:
    1. User starts the add_contact flow.
    2. User digresses the first time.
    3. Bot informs the user that they will continue with the adding a contact and
    then return to the digression.
    4. User digresses a second time and then chooses to continue with
    original flow: add_contact.
    5. Add_contact flow is completed and the bot triggers pattern continue interrupted
    for the checking balance digression.
    6. Bot asks the user if they can do something else.
    """
    sender_id = uuid.uuid4().hex
    processor = calm_handle_digressions_agent.processor

    user_messages = [
        "I want to add a contact",
        "check balance",
        "Loki",
        "check balance",
        "/SetSlots(continue_previous_flow=True)",
        "0712345678",
    ]

    for i, user_msg in enumerate(user_messages):
        await processor.handle_message(UserMessage(user_msg, sender_id=sender_id))
    else:
        actual_responses = []
        tracker = await processor.get_tracker(sender_id)
        for event in tracker.events:
            if isinstance(event, BotUttered):
                actual_responses.append(event.metadata.get("utter_action"))

        assert actual_responses == [
            "utter_ask_contact_name",
            "utter_block_digressions",
            "utter_ask_contact_name",
            "utter_ask_contact_number",
            "utter_ask_continue_previous_flow",
            "utter_block_digressions",
            "utter_ask_contact_number",
            "utter_contact_added",
            "utter_flow_continue_interrupted",
            "utter_check_balance",
            "utter_can_do_something_else",
        ]


async def test_processor_handle_multiple_digressions_continue_with_digression_when_asked(  # noqa: E501
    calm_handle_digressions_agent: Agent,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test how the processor handles multiple duplicate digressions.

    The scenario is as follows:
    1. User starts the add_contact flow.
    2. User digresses the first time.
    3. Bot informs the user that they will continue with the adding a contact and
    then return to the digression.
    4. User digresses a second time and then chooses to continue with
    the digression: check_balance.
    5. Bot triggers the check_balance flow, completes it and then proceeds w
    ith the interrupted flow step: asking for contact_number.
    6. User provides the contact number and the flow is completed.
    """
    sender_id = uuid.uuid4().hex
    processor = calm_handle_digressions_agent.processor

    user_messages = [
        "I want to add a contact",
        "check balance",
        "Loki",
        "check balance",
        "/SetSlots(continue_previous_flow=False)",
        "0712345678",
    ]

    for i, user_msg in enumerate(user_messages):
        await processor.handle_message(UserMessage(user_msg, sender_id=sender_id))
    else:
        actual_responses = []
        tracker = await processor.get_tracker(sender_id)
        for event in tracker.events:
            if isinstance(event, BotUttered):
                actual_responses.append(event.metadata.get("utter_action"))

        assert actual_responses == [
            "utter_ask_contact_name",
            "utter_block_digressions",
            "utter_ask_contact_name",
            "utter_ask_contact_number",
            "utter_ask_continue_previous_flow",
            "utter_continue_interruption",
            "utter_check_balance",
            "utter_flow_continue_interrupted",
            "utter_ask_contact_number",
            "utter_contact_added",
            "utter_can_do_something_else",
        ]
