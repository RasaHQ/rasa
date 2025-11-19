from rasa.agents.constants import (
    A2A_AGENT_CONTEXT_ID_KEY,
    A2A_AGENT_TASK_ID_KEY,
)
from rasa.agents.utils import map_agent_metadata_to_bot_uttered
from rasa.core.constants import (
    ACTIVE_FLOW_METADATA_KEY,
    BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY,
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY,
    BOT_UTTERANCE_AGENT_NAME_KEY,
    BOT_UTTERANCE_AGENT_TASK_ID_KEY,
    BOT_UTTERANCE_CONTEXT_ID_KEY,
    BOT_UTTERANCE_MESSAGE_ID_KEY,
    STEP_ID_METADATA_KEY,
    UTTER_SOURCE_METADATA_KEY,
)


def test_map_agent_metadata_basic_mapping() -> None:
    source = {
        UTTER_SOURCE_METADATA_KEY: "CustomA2AAgent",
        BOT_UTTERANCE_AGENT_NAME_KEY: "my_agent",
        BOT_UTTERANCE_MESSAGE_ID_KEY: "m-123",
        BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY: "intermediate_message",
        BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY: "2023-10-27T10:00:00Z",
        # Protocol-specific identifiers
        A2A_AGENT_TASK_ID_KEY: "task-42",
        A2A_AGENT_CONTEXT_ID_KEY: "ctx-42",
        # Flow context
        ACTIVE_FLOW_METADATA_KEY: "flow-1",
        STEP_ID_METADATA_KEY: "step-1",
    }

    mapped = map_agent_metadata_to_bot_uttered(source)

    assert mapped[UTTER_SOURCE_METADATA_KEY] == "CustomA2AAgent"
    assert mapped[BOT_UTTERANCE_AGENT_NAME_KEY] == "my_agent"
    assert mapped[BOT_UTTERANCE_MESSAGE_ID_KEY] == "m-123"
    assert mapped[BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY] == "intermediate_message"
    assert mapped[BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY] == "2023-10-27T10:00:00Z"

    # Ensure there is no duplication with raw protocol keys
    assert A2A_AGENT_TASK_ID_KEY not in mapped.keys()
    assert A2A_AGENT_CONTEXT_ID_KEY not in mapped.keys()
    assert BOT_UTTERANCE_AGENT_TASK_ID_KEY in mapped.keys()
    assert BOT_UTTERANCE_CONTEXT_ID_KEY in mapped.keys()
    # Protocol identifiers are mapped to agent-prefixed keys
    assert mapped[BOT_UTTERANCE_AGENT_TASK_ID_KEY] == "task-42"
    assert mapped[BOT_UTTERANCE_CONTEXT_ID_KEY] == "ctx-42"

    # Flow context is passed through
    assert mapped[ACTIVE_FLOW_METADATA_KEY] == "flow-1"
    assert mapped[STEP_ID_METADATA_KEY] == "step-1"


def test_map_agent_metadata_skips_none_values() -> None:
    source = {
        UTTER_SOURCE_METADATA_KEY: "A2AAgent",
        BOT_UTTERANCE_AGENT_NAME_KEY: "agent-x",
        BOT_UTTERANCE_MESSAGE_ID_KEY: None,
        BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY: None,
        BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY: None,
        A2A_AGENT_TASK_ID_KEY: None,
        A2A_AGENT_CONTEXT_ID_KEY: None,
        ACTIVE_FLOW_METADATA_KEY: None,
        STEP_ID_METADATA_KEY: None,
    }

    mapped = map_agent_metadata_to_bot_uttered(source)

    # Always present
    assert mapped[UTTER_SOURCE_METADATA_KEY] == "A2AAgent"
    assert mapped[BOT_UTTERANCE_AGENT_NAME_KEY] == "agent-x"

    # None-valued keys are omitted
    assert BOT_UTTERANCE_MESSAGE_ID_KEY not in mapped
    assert BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY not in mapped
    assert BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY not in mapped
    assert BOT_UTTERANCE_AGENT_TASK_ID_KEY not in mapped
    assert BOT_UTTERANCE_CONTEXT_ID_KEY not in mapped
    assert ACTIVE_FLOW_METADATA_KEY not in mapped
    assert STEP_ID_METADATA_KEY not in mapped


def test_map_agent_metadata_partial_input() -> None:
    # Only protocol identifiers provided
    source = {
        A2A_AGENT_TASK_ID_KEY: "t-1",
        A2A_AGENT_CONTEXT_ID_KEY: "c-1",
    }
    mapped = map_agent_metadata_to_bot_uttered(source)

    assert mapped[BOT_UTTERANCE_AGENT_TASK_ID_KEY] == "t-1"
    assert mapped[BOT_UTTERANCE_CONTEXT_ID_KEY] == "c-1"
    # No other keys synthesized
    assert UTTER_SOURCE_METADATA_KEY not in mapped
    assert BOT_UTTERANCE_AGENT_NAME_KEY not in mapped
    assert BOT_UTTERANCE_MESSAGE_ID_KEY not in mapped
    # Ensure no duplication with raw protocol keys
    assert A2A_AGENT_TASK_ID_KEY not in mapped.keys()
    assert A2A_AGENT_CONTEXT_ID_KEY not in mapped.keys()
    assert BOT_UTTERANCE_AGENT_TASK_ID_KEY in mapped.keys()
    assert BOT_UTTERANCE_CONTEXT_ID_KEY in mapped.keys()
