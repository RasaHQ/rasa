import uuid
from typing import Text
from unittest.mock import Mock, patch

import pytest
from aioresponses import aioresponses

from rasa.core.agent import Agent
from rasa.core.channels import CollectingOutputChannel, UserMessage
from rasa.shared.core.events import SlotSet
from rasa.utils.endpoints import EndpointConfig
from tests.conftest import TrainedAsync


@pytest.fixture(scope="session")
async def sender_id() -> str:
    return uuid.uuid4().hex


@pytest.fixture(scope="session")
@patch("langchain_community.vectorstores.faiss.FAISS.from_documents")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_llm_client")
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_embedder")
async def trained_slot_set_bot(
    mock_try_instantiate_llm_client: Mock,
    mock_try_instantiate_embedder: Mock,
    mock_flow_search_create_embedder: Mock,
    mock_from_documents: Mock,
    trained_async: TrainedAsync,
) -> Text:
    mock_try_instantiate_llm_client.return_value = Mock()
    mock_try_instantiate_embedder.return_value = Mock()
    mock_flow_search_create_embedder.return_value = Mock()
    mock_from_documents.return_value = Mock()
    return await trained_async(
        domain="data/test_action_extract_slots_12314/domain.yml",
        config="data/test_action_extract_slots_12314/config.yml",
        training_files=[
            "data/test_action_extract_slots_12314/data/flows.yml",
        ],
        endpoints="data/test_action_extract_slots_12314/endpoints.yml",
    )


@pytest.fixture(scope="session")
def action_server_url() -> str:
    return "https://my-action-server:5055/webhook"


@pytest.fixture
@patch("langchain_community.vectorstores.faiss.FAISS.load_local")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_llm_client")
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_embedder")
async def slot_set_bot_agent(
    mock_try_instantiate_embedder: Mock,
    mock_try_instantiate_llm_client: Mock,
    mock_flow_search_create_embedder: Mock,
    mock_load_local: Mock,
    trained_slot_set_bot: str,
    action_server_url: str,
) -> Agent:
    mock_try_instantiate_embedder.return_value = Mock()
    mock_try_instantiate_llm_client.return_value = Mock()
    mock_flow_search_create_embedder.return_value = Mock()
    mock_load_local.return_value = Mock()
    endpoint = EndpointConfig(action_server_url)
    return Agent.load(model_path=trained_slot_set_bot, action_endpoint=endpoint)


async def test_custom_action_can_set_slot_to_same_value(
    slot_set_bot_agent: Agent,
    action_server_url: str,
    sender_id: str,
) -> None:
    """
    Test that a custom action can set a slot even if the current value
    of the slot is the same as the new value.

    This test verifies the following conversation:
    1. User: "help me install a rsa token"
       Custom action sets rsa_token to "unknown"

    2. User: "hard"
       Custom action sets rsa_token to "hard"

    3. User: "help me install a hard rsa token"
       Custom action sets rsa_token to "hard" (same value)
       This should emit a SlotSet event even though the value is unchanged.

    The key assertion is that in step 3, a SlotSet event is emitted
    even though the slot already has the value "hard".
    """
    output_channel = CollectingOutputChannel()

    # Step 1: First message - set slot to "unknown"
    with aioresponses() as mocked:
        mocked.post(
            action_server_url,
            payload={
                "events": [
                    {
                        "event": "slot",
                        "name": "rsa_token",
                        "value": "unknown",
                    }
                ],
                "responses": [],
            },
        )

        await slot_set_bot_agent.handle_message(
            _build_user_message(
                output_channel, "help me install a rsa token", sender_id
            )
        )

    # Get tracker and verify first SlotSet event
    tracker = await slot_set_bot_agent.tracker_store.get_or_create_tracker(sender_id)
    slot_set_events = [
        e for e in tracker.events if isinstance(e, SlotSet) and e.key == "rsa_token"
    ]
    assert len(slot_set_events) >= 1
    assert slot_set_events[-1].value == "unknown"

    # Step 2: Second message - set slot to "hard"
    with aioresponses() as mocked:
        mocked.post(
            action_server_url,
            payload={
                "events": [
                    {
                        "event": "slot",
                        "name": "rsa_token",
                        "value": "hard",
                    }
                ],
                "responses": [],
            },
        )

        await slot_set_bot_agent.handle_message(
            _build_user_message(output_channel, "hard", sender_id)
        )

    # Verify slot was set to "hard"
    tracker = await slot_set_bot_agent.tracker_store.get_or_create_tracker(sender_id)
    slot_set_events = [
        e for e in tracker.events if isinstance(e, SlotSet) and e.key == "rsa_token"
    ]
    assert len(slot_set_events) >= 2
    assert slot_set_events[-1].value == "hard"
    assert tracker.get_slot("rsa_token") == "hard"

    # Step 3: Third message - try to set slot to "hard" again (same value)
    # The slot already has value "hard", but custom action should still
    # be able to set it to "hard" and emit a SlotSet event
    initial_slot_set_count = len(slot_set_events)

    with aioresponses() as mocked:
        mocked.post(
            action_server_url,
            payload={
                "events": [
                    {
                        "event": "slot",
                        "name": "rsa_token",
                        "value": "hard",
                    }
                ],
                "responses": [],
            },
        )

        await slot_set_bot_agent.handle_message(
            _build_user_message(
                output_channel, "help me install a hard rsa token", sender_id
            )
        )

    # Verify that a SlotSet event was emitted even though the value is the same
    tracker = await slot_set_bot_agent.tracker_store.get_or_create_tracker(sender_id)
    slot_set_events_after = [
        e for e in tracker.events if isinstance(e, SlotSet) and e.key == "rsa_token"
    ]
    # Should have at least one more SlotSet event
    assert len(slot_set_events_after) > initial_slot_set_count, (
        f"Expected more SlotSet events after setting slot to same value. "
        f"Initial count: {initial_slot_set_count}, "
        f"Final count: {len(slot_set_events_after)}"
    )
    # The last SlotSet event should have value "hard"
    assert slot_set_events_after[-1].value == "hard"
    # Verify the slot value in tracker is still "hard"
    assert tracker.get_slot("rsa_token") == "hard"


def _build_user_message(
    output_channel: CollectingOutputChannel, text: str, sender_id: str
):
    return UserMessage(text=text, sender_id=sender_id, output_channel=output_channel)
