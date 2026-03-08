"""Unit tests for MCPTaskAgent."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from rasa.agents.constants import (
    AGENT_METADATA_AGENT_ID_KEY,
    AGENT_METADATA_MODEL_ID_KEY,
    AGENT_METADATA_SENDER_ID_KEY,
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FILLER_MESSAGE,
)
from rasa.agents.protocol.mcp.mcp_task_agent import MCPTaskAgent
from rasa.agents.schemas import AgentInput, AgentInputSlot, AgentToolResult
from rasa.core.available_agents import (
    AgentConfig,
    AgentConfiguration,
    AgentInfo,
    ProtocolConfig,
)
from rasa.shared.constants import (
    DEFAULT_INCLUDE_DATE_TIME,
    DEFAULT_TIMEZONE,
    OPENAI_API_KEY_ENV_VAR,
)
from rasa.shared.core.events import BotUttered, SlotSet
from rasa.shared.exceptions import (
    LLMToolResponseDecodeError,
    ProviderClientAPIException,
)
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMToolCall


class TestMCPTaskAgent:
    """Test cases for MCPTaskAgent."""

    @pytest.fixture
    def mock_agent_input(self) -> AgentInput:
        """Fixture for creating a mock AgentInput."""
        return AgentInput(
            id="test_id",
            user_message="Hello, how can you help me?",
            slots=[
                AgentInputSlot(
                    name="user_name", value="John", type="text", allowed_values=None
                ),
                AgentInputSlot(
                    name="user_age", value=25, type="number", allowed_values=None
                ),
            ],
            conversation_history="Previous conversation...",
            events=[],
            metadata={"key": "value", "nested": {"data": "test"}},
            timestamp="2024-01-15T10:30:00Z",
        )

    @pytest.fixture
    def mcp_task_agent(self, monkeypatch: pytest.MonkeyPatch) -> MCPTaskAgent:
        """Fixture for creating an MCPTaskAgent instance."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in test_mcp_task_agent")
        return MCPTaskAgent.from_config(
            AgentConfig(
                agent=AgentInfo(
                    name="test_task_agent",
                    description="A test task agent for unit testing",
                    protocol=ProtocolConfig.RASA,
                )
            )
        )

    @pytest.fixture
    def mock_output_channel(self):
        """Non-streaming output channel fixture (LLM called via acompletion)."""
        channel = MagicMock()
        channel.supports_streaming = False
        channel.send_text_message = AsyncMock()
        return channel

    def test_render_prompt_template_basic_rendering(
        self, mcp_task_agent: MCPTaskAgent, mock_agent_input: AgentInput
    ):
        """Test basic prompt template rendering with all context variables."""

        mock_now = datetime(2024, 1, 15, 14, 30, 45, tzinfo=ZoneInfo("UTC"))
        with patch(
            "rasa.shared.utils.datetime_utils.get_current_datetime"
        ) as mock_get_current_datetime:
            mock_get_current_datetime.return_value = mock_now

            result = mcp_task_agent.render_prompt_template(mock_agent_input)

            # Verify the template was rendered with correct date/time values
            assert "- Current date: 15 January, 2024" in result  # current_date
            assert "- Current time: 14:30:45 (UTC)" in result  # current_time
            assert "- Current day: Monday" in result  # current_day

            # Verify other context variables are included
            assert "A test task agent for unit testing" in result  # description

            # Verify template structure is maintained (MCP Task Agent template)
            assert "### Date & Time Context" in result
            assert "### Description of your capabilities" in result
            assert "### Task" in result
            assert "### Instructions" in result

    def test_render_prompt_template_excludes_specified_fields(
        self, mcp_task_agent: MCPTaskAgent, mock_agent_input: AgentInput
    ):
        """Test that render_prompt_template excludes some fields from context."""
        with patch(
            "rasa.shared.utils.datetime_utils.get_current_datetime"
        ) as mock_get_current_datetime:
            mock_get_current_datetime.return_value = datetime(
                2024, 1, 15, 14, 30, 45, tzinfo=ZoneInfo("UTC")
            )

            result = mcp_task_agent.render_prompt_template(mock_agent_input)

            # Verify excluded fields are not in the rendered template
            assert "test_id" not in result  # id should be excluded
            assert "2024-01-15T10:30:00Z" not in result  # timestamp should be excluded

    def test_mcp_task_agent_is_exit_conditions_met_empty_input(self):
        exit_met, internal_error = MCPTaskAgent._is_exit_conditions_met(
            agent_input=AgentInput(
                id="test",
                user_message="test",
                slots=[],
                conversation_history="test",
                events=[],
                metadata={},
            ),
            slots={},
        )
        assert exit_met is False
        assert internal_error is None

    def test_mcp_task_agent_is_exit_conditions_met_no_exit_conditions(self):
        """Test that when there are no exit conditions, it returns True
        (no conditions to check).
        """
        exit_met, internal_error = MCPTaskAgent._is_exit_conditions_met(
            agent_input=AgentInput(
                id="test",
                user_message="test",
                slots=[],
                conversation_history="test",
                events=[],
                metadata={"exit_if": []},  # Empty list of exit conditions
            ),
            slots={"some_slot": "some_value"},
        )
        assert exit_met is True  # No conditions to check, so all are met
        assert internal_error is None

    def test_mcp_task_agent_is_exit_conditions_met_empty_slots(self):
        """Test that when slots is empty, it returns False early."""
        exit_met, internal_error = MCPTaskAgent._is_exit_conditions_met(
            agent_input=AgentInput(
                id="test",
                user_message="test",
                slots=[],
                conversation_history="test",
                events=[],
                metadata={"exit_if": ["slots.test == 'value'"]},
            ),
            slots={},  # Empty slots
        )
        assert exit_met is False  # Early return when slots is empty
        assert internal_error is None

    @pytest.mark.parametrize(
        "slots, exit_conditions, expected_exit_met",
        [
            # Basic string comparison tests
            ({}, ["slots.test == 'some_value'"], False),
            ({"test": "some_value"}, ["slots.test == 'some_value'"], True),
            ({"test": "some_other_value"}, ["slots.test == 'some_value'"], False),
            (
                {"test": "some_value", "test2": "some_other_value"},
                ["slots.test == 'some_value'"],
                True,
            ),
            (
                {"test": "some_value", "test2": "some_other_value"},
                ["slots.test == 'some_value' and slots.test2 == 'some_other_value'"],
                True,
            ),
            (
                {"test": "some_value", "test2": "different_value"},
                ["slots.test == 'some_value' and slots.test2 == 'some_other_value'"],
                False,
            ),
            # Boolean value tests
            ({}, ["slots.flag == True"], False),
            ({"flag": True}, ["slots.flag == True"], True),
            ({"flag": False}, ["slots.flag == True"], False),
            ({"flag": "true"}, ["slots.flag == True"], False),  # String vs boolean
            (
                {"flag": True, "status": "active"},
                ["slots.flag == True and slots.status == 'active'"],
                True,
            ),
            (
                {"flag": True, "status": "inactive"},
                ["slots.flag == True and slots.status == 'active'"],
                False,
            ),
            # Numeric comparison tests
            ({"count": 5}, ["slots.count == 5"], True),
            ({"count": 5}, ["slots.count > 3"], True),
            ({"count": 5}, ["slots.count < 10"], True),
            ({"count": 5}, ["slots.count >= 5"], True),
            ({"count": 5}, ["slots.count <= 5"], True),
            ({"count": 5}, ["slots.count != 3"], True),
            # Multiple conditions with mixed types
            (
                {"name": "John", "age": 30, "active": True},
                ["slots.name == 'John'", "slots.age > 25", "slots.active == True"],
                True,
            ),
            (
                {"name": "John", "age": 20, "active": True},
                ["slots.name == 'John'", "slots.age > 25", "slots.active == True"],
                False,
            ),
            # Edge cases
            ({"empty": ""}, ["slots.empty == ''"], True),  # Empty string comparison
            ({"zero": 0}, ["slots.zero == 0"], True),
            ({"none": None}, ["slots.none == null"], True),
            ({"false": False}, ["slots.false == False"], True),
            # Complex boolean expressions
            (
                {"status": "pending"},
                ["slots.status == 'pending' or slots.status == 'active'"],
                True,
            ),
            (
                {"status": "active"},
                ["slots.status == 'pending' or slots.status == 'active'"],
                True,
            ),
            (
                {"status": "completed"},
                ["slots.status == 'pending' or slots.status == 'active'"],
                False,
            ),
            (
                {"count": 5, "active": True},
                [
                    "slots.count > 3 and (slots.active == True or slots.status == 'active')"  # noqa: E501
                ],
                True,
            ),
            (
                {"count": 1, "active": False},
                [
                    "slots.count > 3 and (slots.active == True or slots.status == 'active')"  # noqa: E501
                ],
                False,
            ),
            # Non-existent slot references
            (
                {"existing": "value"},
                ["slots.non_existent == 'something'"],
                False,
            ),  # Slot doesn't exist
            (
                {"existing": "value"},
                ["slots.existing == 'value' and slots.non_existent == 'something'"],
                False,
            ),  # Mixed existing/non-existing
            (
                {"existing": "value"},
                ["slots.existing == 'value' or slots.non_existent == 'something'"],
                True,
            ),  # OR with non-existing (first condition succeeds)
            (
                {"existing": "value"},
                ["slots.non_existent == 'something' or slots.existing == 'value'"],
                True,
            ),  # OR with non-existing (second condition succeeds)
        ],
    )
    def test_mcp_task_agent_is_exit_conditions_met_with_exit_conditions(
        self, slots: Dict[str, Any], exit_conditions: List[str], expected_exit_met: bool
    ):
        exit_met, internal_error = MCPTaskAgent._is_exit_conditions_met(
            agent_input=AgentInput(
                id="test",
                user_message="test",
                slots=[],
                conversation_history="test",
                events=[],
                metadata={"exit_if": exit_conditions},
            ),
            slots=slots,
        )
        assert exit_met is expected_exit_met
        assert internal_error is None

    def test_mcp_task_agent_is_exit_conditions_met_with_internal_error(self):
        exit_met, internal_error = MCPTaskAgent._is_exit_conditions_met(
            agent_input=AgentInput(
                id="test",
                user_message="test",
                slots=[],
                conversation_history="test",
                events=[],
                metadata={"exit_if": ["slots.test == slots !!!='some_other_value'"]},
            ),
            slots={"test": "some_value"},
        )
        assert exit_met is False
        assert internal_error is not None

    @pytest.mark.parametrize(
        "metadata, expected_slot_names",
        [
            # Test with valid exit conditions
            (
                {"exit_if": ["slots.user_name == 'John'", "slots.user_age > 25"]},
                ["user_name", "user_age"],
            ),
            # Test with no exit conditions
            ({}, []),
            # Test with invalid slots (should only return existing slots)
            (
                {
                    "exit_if": [
                        "slots.non_existent == 'value'",
                        "slots.user_name == 'John'",
                    ]
                },
                ["user_name"],
            ),
        ],
    )
    def test_get_slot_names_from_exit_conditions(
        self, mock_agent_input, metadata, expected_slot_names
    ):
        """Test extracting slot names from various exit conditions."""
        mock_agent_input.metadata = metadata

        slot_names = MCPTaskAgent._get_slot_names_from_exit_conditions(mock_agent_input)

        assert len(slot_names) == len(expected_slot_names)
        for expected_slot in expected_slot_names:
            assert expected_slot in slot_names

    @pytest.mark.parametrize(
        (
            "slot_name, slot_type, slot_value, allowed_values, "
            "expected_tool_name, expected_description_parts"
        ),
        [
            # Text slot
            (
                "user_name",
                "text",
                "John",
                None,
                "set_slot_user_name",
                ["Set the slot 'user_name'", "text"],
            ),
            # Categorical slot
            (
                "category",
                "categorical",
                "A",
                ["A", "B", "C"],
                "set_slot_category",
                ["categorical", "['A', 'B', 'C']"],
            ),
        ],
    )
    def test_get_slot_specific_set_slot_tool(
        self,
        slot_name,
        slot_type,
        slot_value,
        allowed_values,
        expected_tool_name,
        expected_description_parts,
    ):
        """Test creating set slot tool for different slot types."""
        slot = AgentInputSlot(
            name=slot_name,
            value=slot_value,
            type=slot_type,
            allowed_values=allowed_values,
        )

        tool = MCPTaskAgent.get_slot_specific_set_slot_tool(slot)

        assert tool["type"] == "function"
        assert tool["function"]["name"] == expected_tool_name
        assert "slot_value" in tool["function"]["parameters"]["properties"]
        assert tool["function"]["parameters"]["required"] == ["slot_value"]

        description = tool["function"]["description"]
        for part in expected_description_parts:
            assert part in description

    @pytest.mark.parametrize(
        "tool_name, expected_slot_name",
        [
            ("set_slot_user_name", "user_name"),  # Valid tool name
            ("invalid_tool_name", None),  # Invalid tool name
            ("some_other_tool", None),  # Tool name that doesn't match pattern
        ],
    )
    def test_get_slot_name_from_tool_name(
        self, mcp_task_agent, tool_name, expected_slot_name
    ):
        """Test extracting slot name from various tool names."""
        slot_name = mcp_task_agent._get_slot_name_from_tool_name(tool_name)

        assert slot_name == expected_slot_name

    @pytest.mark.parametrize(
        "slot_value, expected_result",
        [
            ("test_value", "test_value"),  # String value
            ("true", True),  # Boolean string true
            ("false", False),  # Boolean string false
            ("TRUE", True),  # Boolean string case insensitive
            ("maybe", "maybe"),  # Non-boolean string
            (123, 123),  # Non-string value
        ],
    )
    def test_run_set_slot_tool_various_values(
        self, mcp_task_agent, slot_value, expected_result
    ):
        """Test running set slot tool with various value types."""
        result = mcp_task_agent._run_set_slot_tool(
            "test_slot", {"slot_value": slot_value}
        )

        assert result == {"test_slot": expected_result}

    @pytest.mark.parametrize(
        "initial_slot_values, current_slot_values, metadata, expected_events",
        [
            # Exit-condition slot changed -> one SlotSet event
            (
                {"user_name": "John", "user_age": 25},
                {"user_name": "Jane", "user_age": 25},
                {"exit_if": ["slots.user_name == 'Jane'"]},
                [SlotSet("user_name", "Jane")],
            ),
            # No change -> empty list
            (
                {"user_name": "John"},
                {"user_name": "John"},
                {"exit_if": ["slots.user_name == 'John'"]},
                [],
            ),
            # No exit conditions -> empty list
            (
                {"user_name": "John"},
                {"user_name": "Jane"},
                {},
                [],
            ),
            # Multiple exit-condition slots, only one changed
            (
                {"user_name": "John", "user_age": 25},
                {"user_name": "Jane", "user_age": 25},
                {
                    "exit_if": [
                        "slots.user_name == 'Jane'",
                        "slots.user_age > 20",
                    ]
                },
                [SlotSet("user_name", "Jane")],
            ),
            # Both exit-condition slots changed
            (
                {"user_name": "John", "user_age": 25},
                {"user_name": "Jane", "user_age": 30},
                {
                    "exit_if": [
                        "slots.user_name == 'Jane'",
                        "slots.user_age > 25",
                    ]
                },
                [SlotSet("user_name", "Jane"), SlotSet("user_age", 30)],
            ),
        ],
    )
    def test_get_slot_set_events_for_changed_slots(
        self,
        mcp_task_agent: MCPTaskAgent,
        mock_agent_input: AgentInput,
        initial_slot_values: Dict[str, Any],
        current_slot_values: Dict[str, Any],
        metadata: Dict[str, Any],
        expected_events: List[SlotSet],
    ):
        """Test that only exit-condition slots that changed produce SlotSet events."""
        mock_agent_input.metadata = metadata
        result = mcp_task_agent._get_slot_set_events_for_changed_slots(
            mock_agent_input, initial_slot_values, current_slot_values
        )
        assert len(result) == len(expected_events)
        result_keys = {e.key for e in result}
        result_values = {e.key: e.value for e in result}
        for expected in expected_events:
            assert expected.key in result_keys
            assert result_values[expected.key] == expected.value

    def test_generate_agent_task_completed_output(
        self, mcp_task_agent, mock_agent_input
    ):
        """Test generating task completed output."""
        slots = {"user_name": "John", "user_age": 25}
        tool_results = {
            "call_1": AgentToolResult(
                tool_name="test_tool",
                result="test_result",
                is_error=False,
            )
        }
        mock_agent_input.metadata = {"exit_if": ["slots.user_name == 'John'"]}

        with patch.object(
            mcp_task_agent, "_get_structured_results_for_agent_output"
        ) as mock_get_results:
            mock_get_results.return_value = [
                [{"name": "test_tool", "result": "test_result"}]
            ]

            result = mcp_task_agent._generate_agent_task_completed_output(
                mock_agent_input, slots, tool_results
            )

            assert result.id == mock_agent_input.id
            assert result.status.name == "COMPLETED"
            assert len(result.events) == 1
            assert result.events[0].key == "user_name"
            assert result.events[0].value == "John"
            mock_get_results.assert_called_once_with(mock_agent_input, tool_results)

    def test_generate_agent_task_completed_output_no_matching_slots(
        self, mcp_task_agent, mock_agent_input
    ):
        """Test generating task completed output with no matching slots."""
        slots = {"other_slot": "value"}
        tool_results = {}
        mock_agent_input.metadata = {"exit_if": ["slots.user_name == 'John'"]}

        with patch.object(
            mcp_task_agent, "_get_structured_results_for_agent_output"
        ) as mock_get_results:
            mock_get_results.return_value = []

            result = mcp_task_agent._generate_agent_task_completed_output(
                mock_agent_input, slots, tool_results
            )

            assert result.id == mock_agent_input.id
            assert result.status.name == "COMPLETED"
            assert len(result.events) == 0  # No matching slots

    def test_generate_agent_task_completed_output_deduplicates_slot_set_events(
        self, mcp_task_agent, mock_agent_input
    ):
        """Avoid duplicate SlotSet events when additional events already include one."""
        slots = {"user_name": "John"}
        tool_results = {}
        mock_agent_input.metadata = {"exit_if": ["slots.user_name == 'John'"]}
        additional_events = [SlotSet("user_name", "John"), BotUttered(text="progress")]

        with patch.object(
            mcp_task_agent, "_get_structured_results_for_agent_output"
        ) as mock_get_results:
            mock_get_results.return_value = []

            result = mcp_task_agent._generate_agent_task_completed_output(
                mock_agent_input,
                slots,
                tool_results,
                additional_events=additional_events,
            )

        assert result.status.name == "COMPLETED"
        assert result.events is not None
        slot_events = [event for event in result.events if isinstance(event, SlotSet)]
        assert len(slot_events) == 1
        assert slot_events[0].key == "user_name"
        assert slot_events[0].value == "John"
        assert any(
            isinstance(event, BotUttered) and event.text == "progress"
            for event in result.events
        )

    def test_check_exit_conditions_internal_error_sets_response_message(
        self, mcp_task_agent, mock_agent_input
    ):
        """Internal exit-condition errors should keep a user-facing response message."""
        mock_agent_input.metadata = {"exit_if": ["slots.user_name == 'John'"]}
        initial_slot_values = {"user_name": "John", "user_age": 25}
        current_slot_values = {"user_name": "John", "user_age": 25}

        with patch.object(
            mcp_task_agent,
            "_is_exit_conditions_met",
            return_value=(False, "mock exit-if eval error"),
        ):
            result = mcp_task_agent._check_exit_conditions_after_iteration(
                agent_input=mock_agent_input,
                initial_slot_values=initial_slot_values,
                current_slot_values=current_slot_values,
                tool_results={},
                generated_events=[],
                accumulated_tool_output_events=[],
            )

        assert result is not None
        assert result.status.name == "FATAL_ERROR"
        assert (
            result.response_message
            == "An internal error occurred while checking the exit conditions."
        )
        assert result.error_message == "mock exit-if eval error"

    def test_render_prompt_template_with_slot_names(
        self, mcp_task_agent, mock_agent_input
    ):
        """Test rendering prompt template with slot names."""
        mock_agent_input.metadata = {"exit_if": ["slots.user_name == 'John'"]}

        with patch(
            "rasa.shared.utils.datetime_utils.get_current_datetime"
        ) as mock_get_current_datetime:
            mock_get_current_datetime.return_value = datetime(
                2024, 1, 15, 14, 30, 45, tzinfo=ZoneInfo("UTC")
            )

            result = mcp_task_agent.render_prompt_template(mock_agent_input)

            assert "user_name" in result  # slot_names should be included
            assert "- Current date: 15 January, 2024" in result
            assert "- Current time: 14:30:45 (UTC)" in result
            assert "- Current day: Monday" in result

    @pytest.mark.parametrize(
        "slots, metadata, expected_assertions",
        [
            # Test basic slot access with slot_names
            (
                [
                    AgentInputSlot(
                        name="user_name",
                        value="John",
                        type="text",
                    ),
                    AgentInputSlot(
                        name="user_age",
                        value=25,
                        type="float",
                    ),
                ],
                {"exit_if": ["slots.user_name == 'John'"]},
                [
                    ("user_name=John", True),
                    ("user_age=25", True),
                    ("Direct: John, 25", True),
                    ("user_name", True),  # from slot_names
                ],
            ),
            # Test None values are excluded
            (
                [
                    AgentInputSlot(
                        name="user_name",
                        value="John",
                        type="text",
                    ),
                    AgentInputSlot(
                        name="user_age",
                        value=None,
                        type="float",
                    ),
                    AgentInputSlot(
                        name="user_email",
                        value="john@example.com",
                        type="text",
                    ),
                ],
                {},
                [
                    ("user_name=John", True),
                    ("user_email=john@example.com", True),
                    ("user_age=None", False),  # Should not appear
                    ("Has user_age: no", True),  # Should not be in dict
                ],
            ),
            # Test both slots and slot_names accessible together
            (
                [
                    AgentInputSlot(
                        name="user_name",
                        value="John",
                        type="text",
                    ),
                    AgentInputSlot(
                        name="user_age",
                        value=25,
                        type="float",
                    ),
                ],
                {"exit_if": ["slots.user_name == 'John'"]},
                [
                    ("user_name=John", True),
                    ("user_age=25", True),
                    ("user_name", True),  # from slot_names
                    ("Direct: John", True),
                ],
            ),
        ],
    )
    def test_render_prompt_template_slots_access(
        self,
        mcp_task_agent: MCPTaskAgent,
        slots: List[AgentInputSlot],
        metadata: Dict[str, Any],
        expected_assertions: List[Tuple[str, bool]],
    ):
        """Test that slots, slot_names are accessible in the task agent prompt."""
        # Single comprehensive template that covers all test cases
        template = (
            "User message: {{user_message}}\n"
            "Slots count: {{ slots|length }}\n"
            "Slots: {% for slot_name, slot_value in slots.items() %}"
            "{{ slot_name }}={{ slot_value }}"
            "{% endfor %}\n"
            "Direct: {{ slots.user_name if slots.user_name else 'not set' }}, "
            "{{ slots.user_age if slots.user_age else 'not set' }}\n"
            "Has user_name: {{ 'yes' if 'user_name' in slots else 'no' }}\n"
            "Has user_age: {{ 'yes' if 'user_age' in slots else 'no' }}\n"
            "Slot names: {{ slot_names }}"
        )
        mcp_task_agent.prompt_template = template

        agent_input = AgentInput(
            id="test_id",
            user_message="Hello",
            slots=slots,
            conversation_history="",
            events=[],
            metadata=metadata,
            timestamp="2024-01-15T10:30:00Z",
        )

        with patch(
            "rasa.shared.utils.datetime_utils.get_current_datetime"
        ) as mock_get_current_datetime:
            mock_get_current_datetime.return_value = datetime(
                2024, 1, 15, 14, 30, 45, tzinfo=ZoneInfo("UTC")
            )

            result = mcp_task_agent.render_prompt_template(agent_input)

            # Verify all expected assertions
            for expected_text, should_be_present in expected_assertions:
                if should_be_present:
                    assert (
                        expected_text in result
                    ), f"Expected '{expected_text}' in result"
                else:
                    assert (
                        expected_text not in result
                    ), f"Expected '{expected_text}' NOT in result"

    @pytest.mark.parametrize(
        "include_date_time, timezone, expected_datetime_present, expected_date_format,"
        "expected_time_format, expected_day",
        [
            # include_date_time is True (default), should include datetime
            (
                DEFAULT_INCLUDE_DATE_TIME,
                DEFAULT_TIMEZONE,
                True,
                "15 January, 2024",
                "14:30:45",
                "Monday",
            ),
            # include_date_time is True with custom timezone
            (
                True,
                "America/New_York",
                True,
                "15 January, 2024",
                "14:30:45",
                "Monday",
            ),
            # include_date_time is False
            # should NOT include datetime
            (False, DEFAULT_TIMEZONE, False, None, None, None),
            # include_date_time is False with custom timezone
            # should NOT include datetime
            (False, "America/New_York", False, None, None, None),
        ],
    )
    def test_render_prompt_template_includes_current_datetime_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_agent_input: AgentInput,
        include_date_time: bool,
        timezone: str,
        expected_datetime_present: bool,
        expected_date_format: Optional[str],
        expected_time_format: Optional[str],
        expected_day: Optional[str],
    ):
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in test_mcp_task_agent")

        # Create agent with datetime configuration
        agent_config = AgentConfig(
            agent=AgentInfo(
                name="test_task_agent",
                description="A test task agent for unit testing",
                protocol=ProtocolConfig.RASA,
            ),
            configuration=AgentConfiguration(
                include_date_time=include_date_time,
                timezone=timezone,
            ),
        )
        mcp_task_agent = MCPTaskAgent.from_config(agent_config)

        # Mock get_current_datetime to return a fixed datetime
        mock_now = datetime(2024, 1, 15, 14, 30, 45, tzinfo=ZoneInfo(timezone))
        with patch(
            "rasa.shared.utils.datetime_utils.get_current_datetime"
        ) as mock_get_current_datetime:
            mock_get_current_datetime.return_value = mock_now

            result = mcp_task_agent.render_prompt_template(mock_agent_input)

            if expected_datetime_present:
                # Verify datetime section is present
                assert "### Date & Time Context" in result
                assert expected_date_format in result
                assert expected_time_format in result
                assert expected_day in result
                assert mock_now.tzname() in result
                # Verify get_current_datetime was called
                mock_get_current_datetime.assert_called_once_with(timezone=timezone)
            else:
                # Verify datetime section is NOT present
                assert "### Date & Time Context" not in result
                assert "Current date:" not in result
                assert "Current time:" not in result
                assert "Current day:" not in result
                # Verify get_current_datetime was NOT called
                mock_get_current_datetime.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "llm_response, expected_status, expected_message_keyword",
        [
            # No/empty response (no choices, no tool_calls) → RECOVERABLE_ERROR
            (
                LLMResponse(
                    id="test_id", created=1642248600, choices=[], tool_calls=None
                ),
                "RECOVERABLE_ERROR",
                "No response from LLM",
            ),
            # Content only with exactly one choice (len(choices)==1) → INPUT_REQUIRED
            (
                LLMResponse(
                    id="test_id",
                    created=1642248600,
                    choices=["Test response"],
                    tool_calls=[],
                ),
                "INPUT_REQUIRED",
                "Test response",
            ),
            # Whitespace or empty string → INPUT_REQUIRED, no BotUttered
            (
                LLMResponse(
                    id="test_id",
                    created=1642248600,
                    choices=[""],
                    tool_calls=[],
                ),
                "INPUT_REQUIRED",
                None,  # no bot utterance expected
            ),
            (
                LLMResponse(
                    id="test_id",
                    created=1642248600,
                    choices=["  "],
                    tool_calls=[],
                ),
                "INPUT_REQUIRED",
                None,  # no bot utterance expected
            ),
        ],
    )
    async def test_send_message_llm_response_scenarios(
        self,
        mcp_task_agent,
        mock_agent_input,
        mock_output_channel,
        llm_response,
        expected_status,
        expected_message_keyword,
    ):
        """Test send_message with various LLM response scenarios.

        The content-only path (INPUT_REQUIRED) requires exactly one choice
        (len(llm_response.choices) == 1) and no tool calls.
        Whitespace or empty content yields INPUT_REQUIRED but no BotUttered.
        """
        mock_agent_input.recipient_id = "test_user"
        with patch.object(mcp_task_agent, "llm_client") as mock_llm_client:
            mock_llm_client.acompletion = AsyncMock(return_value=llm_response)

            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

            assert result.id == mock_agent_input.id
            assert result.status.name == expected_status
            if expected_status == "RECOVERABLE_ERROR":
                assert expected_message_keyword in result.error_message
            elif expected_status == "INPUT_REQUIRED":
                assert result.events is not None
                bot_utters = [e for e in result.events if isinstance(e, BotUttered)]
                if expected_message_keyword is None:
                    # Whitespace/empty: no bot utterance
                    assert len(bot_utters) == 0
                else:
                    assert any(
                        expected_message_keyword in (e.text or "") for e in bot_utters
                    )
            else:
                assert result.response_message == expected_message_keyword

    @pytest.mark.asyncio
    async def test_send_message_tool_not_available(
        self, mcp_task_agent, mock_agent_input, mock_output_channel
    ):
        """Test send_message with unavailable tool."""
        from rasa.shared.providers.llm.llm_response import LLMResponse, LLMToolCall

        mock_agent_input.recipient_id = "test_user"
        mock_llm_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool call attempted"],
            tool_calls=[
                LLMToolCall(
                    id="call_123",
                    type="function",
                    tool_name="unavailable_tool",
                    tool_args={},
                )
            ],
        )

        with (
            patch.object(mcp_task_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
        ):
            mock_llm_client.acompletion = AsyncMock(return_value=mock_llm_response)
            mock_get_tools.return_value = []  # No available tools

            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

            assert result.id == mock_agent_input.id
            assert result.status.name == "FATAL_ERROR"
            assert "Tool unavailable_tool is not available" in result.error_message

    @pytest.mark.asyncio
    async def test_send_message_set_slot_tool_success(
        self, mcp_task_agent, mock_agent_input, mock_output_channel
    ):
        """Test send_message with successful set slot tool call.

        Exit conditions are evaluated in-loop and should complete immediately
        once the slot-setting tool satisfies them, without an extra LLM turn.
        """
        mock_agent_input.recipient_id = "test_user"
        mock_agent_input.metadata = {"exit_if": ["slots.user_name == 'Jane'"]}
        mock_tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="set_slot_user_name",
            tool_args={"slot_value": "Jane"},
        )

        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Slot set successfully"],
            tool_calls=[mock_tool_call],
        )

        with (
            patch.object(mcp_task_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
        ):
            mock_llm_client.acompletion = AsyncMock(return_value=first_response)
            mock_tool = MagicMock()
            mock_tool.name = "set_slot_user_name"
            mock_get_tools.return_value = [mock_tool]

            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

            assert result.id == mock_agent_input.id
            assert result.status.name == "COMPLETED"
            assert result.response_message is None
            assert result.events is not None
            slot_events = [e for e in result.events if isinstance(e, SlotSet)]
            assert len(slot_events) == 1
            assert slot_events[0].key == "user_name"
            assert slot_events[0].value == "Jane"
            assert mock_llm_client.acompletion.call_count == 1

    @pytest.mark.asyncio
    async def test_send_message_filler_message_in_agent_output_events(
        self, mcp_task_agent, mock_agent_input
    ):
        """Agent output events include BotUttered for filler message before tool run.

        When send_message runs with an output_channel and the LLM returns
        tool_calls with content (filler), the filler is sent and a BotUttered
        event is appended to generated_events. The final AgentOutput.events
        must include that filler event (together with any slot events).
        """
        mock_agent_input.recipient_id = "user_1"
        mock_agent_input.metadata = {
            AGENT_METADATA_AGENT_ID_KEY: "agent_1",
            AGENT_METADATA_MODEL_ID_KEY: "gpt-4o",
            "exit_if": ["slots.user_name == 'Jane'"],
        }
        mock_channel = MagicMock()
        mock_channel.supports_streaming = False
        mock_channel.send_text_message = AsyncMock()

        filler_text = "Let me check that for you."
        mock_tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="set_slot_user_name",
            tool_args={"slot_value": "Jane"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=[filler_text],
            tool_calls=[mock_tool_call],
        )
        with (
            patch.object(mcp_task_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
        ):
            mock_llm_client.acompletion = AsyncMock(return_value=first_response)
            mock_tool = MagicMock()
            mock_tool.name = "set_slot_user_name"
            mock_get_tools.return_value = [mock_tool]

            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_channel
            )

        assert result.id == mock_agent_input.id
        assert result.events is not None
        filler_events = [
            e
            for e in result.events
            if isinstance(e, BotUttered)
            and e.metadata.get("agent_message_type")
            == BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FILLER_MESSAGE
        ]
        assert len(filler_events) == 1
        assert filler_events[0].text == filler_text
        # Slot change from set_slot should also be present
        slot_events = [e for e in result.events if isinstance(e, SlotSet)]
        assert len(slot_events) == 1
        assert slot_events[0].key == "user_name"
        assert slot_events[0].value == "Jane"
        assert mock_llm_client.acompletion.call_count == 1

    @pytest.mark.asyncio
    async def test_send_message_set_slot_tool_slot_not_found(
        self, mcp_task_agent, mock_agent_input, mock_output_channel
    ):
        """Test send_message with set slot tool for non-existent slot."""
        mock_agent_input.recipient_id = "test_user"
        mock_tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="set_slot_non_existent",
            tool_args={"slot_value": "value"},
        )

        mock_llm_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Attempting to set slot"],
            tool_calls=[mock_tool_call],
        )

        with (
            patch.object(mcp_task_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
        ):
            mock_llm_client.acompletion = AsyncMock(return_value=mock_llm_response)
            mock_tool = MagicMock()
            mock_tool.name = "set_slot_non_existent"
            mock_get_tools.return_value = [mock_tool]

            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

            assert result.id == mock_agent_input.id
            assert result.status.name == "FATAL_ERROR"
            assert "not found in agent input" in result.error_message

    @pytest.mark.asyncio
    async def test_send_message_malformed_tool_response_retry(
        self, mcp_task_agent, mock_agent_input, mock_output_channel
    ):
        """Test send_message with malformed tool response that triggers retry."""
        mock_agent_input.recipient_id = "test_user"
        # Create a proper exception with original_exception attribute
        decode_error = LLMToolResponseDecodeError("Invalid JSON")
        provider_exception = ProviderClientAPIException("Decode error")
        provider_exception.original_exception = decode_error

        mock_llm_client = MagicMock()
        mock_llm_client.acompletion = AsyncMock(
            side_effect=[
                provider_exception,
                LLMResponse(
                    id="test_id",
                    created=1642248600,
                    choices=["Success response"],
                    tool_calls=[],
                ),
            ]
        )
        mcp_task_agent.llm_client = mock_llm_client

        with (
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
            patch.object(
                mcp_task_agent,
                "build_messages_for_llm_request",
                wraps=mcp_task_agent.build_messages_for_llm_request,
            ) as mock_build_messages,
        ):
            mock_get_tools.return_value = []

            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

            assert result.id == mock_agent_input.id
            assert result.status.name == "INPUT_REQUIRED"
            # When output_channel is set, response_message is omitted; check events
            assert result.events is not None
            bot_utters = [e for e in result.events if isinstance(e, BotUttered)]
            assert any("Success response" in (e.text or "") for e in bot_utters)
            # We want to see if the second LLM call is made.
            assert mock_llm_client.acompletion.call_count == 2

            # Check that the 2nd call includes the system message
            # for malformed tool response
            second_call_args = mock_llm_client.acompletion.call_args_list[1]
            messages = second_call_args[0][0]  # First positional argument

            # Should have 3 messages
            # system, user, and malformed tool response system message
            assert len(messages) == 3
            assert messages[0]["role"] == "system"  # Original system message
            assert messages[1]["role"] == "user"  # User message
            assert messages[2]["role"] == "system"  # Malformed tool response message

            # System message for malformed tool response
            system_message = (
                "The previous tool response contained invalid or incomplete JSON and "
                "could not be parsed. Retry by generating a tool response in STRICT "
                "JSON string format only. Ensure the JSON is fully well-formed and "
                "corresponds exactly to the user's last request."
            )
            assert messages[2]["content"] == system_message
            assert mock_build_messages.call_count == 1

    @pytest.mark.asyncio
    async def test_send_message_max_iterations_reached(
        self, mcp_task_agent, mock_agent_input, mock_output_channel
    ):
        """Test send_message when max iterations are reached."""
        # Create a tool call that will keep the agent in a loop
        mock_tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )

        mock_llm_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed"],
            tool_calls=[mock_tool_call],
        )

        mock_tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )

        mock_agent_input.recipient_id = "test_user"
        with (
            patch.object(mcp_task_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_task_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_task_agent, "_is_exit_conditions_met"
            ) as mock_exit_conditions,
        ):
            # Always return the same response to create an infinite loop
            mock_llm_client.acompletion = AsyncMock(return_value=mock_llm_response)
            mock_tool = MagicMock()
            mock_tool.name = "other_tool"
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = mock_tool_output
            # Ensure exit conditions are NOT met so we hit max iterations
            mock_exit_conditions.return_value = (False, None)

            # Set max iterations to 1 to force completion
            mcp_task_agent.MAX_ITERATIONS = 1

            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

            assert result.id == mock_agent_input.id
            assert result.status.name == "COMPLETED"
            # Check that response_message is not None and contains the expected text
            assert result.response_message is not None
            assert "couldn't provide a final answer" in result.response_message

    @pytest.mark.asyncio
    async def test_send_message_max_iterations_includes_changed_slot_events(
        self, mcp_task_agent, mock_agent_input, mock_output_channel
    ):
        """Test that when max iterations is reached, output includes slot changes."""
        mock_agent_input.recipient_id = "test_user"
        mock_agent_input.metadata = {"exit_if": ["slots.user_name == 'Jane'"]}
        set_slot_call = LLMToolCall(
            id="call_set_slot",
            type="function",
            tool_name="set_slot_user_name",
            tool_args={"slot_value": "Jane"},
        )
        other_tool_call = LLMToolCall(
            id="call_other",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Setting slot"],
            tool_calls=[set_slot_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Calling other tool"],
            tool_calls=[other_tool_call],
        )
        mock_tool_output = AgentToolResult(
            tool_name="other_tool",
            result="ok",
            is_error=False,
        )

        with (
            patch.object(mcp_task_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
            patch.object(
                mcp_task_agent, "_execute_tool_call", new_callable=AsyncMock
            ) as mock_execute_tool,
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            set_slot_tool = MagicMock()
            set_slot_tool.name = "set_slot_user_name"
            other_tool = MagicMock()
            other_tool.name = "other_tool"
            mock_get_tools.return_value = [set_slot_tool, other_tool]
            mock_execute_tool.return_value = mock_tool_output

            mcp_task_agent.MAX_ITERATIONS = 2
            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

            assert result.id == mock_agent_input.id
            assert result.status.name == "COMPLETED"
            assert result.response_message is None
            # Slot set in first iteration must be preserved in output
            assert result.events is not None
            slot_events = [e for e in result.events if isinstance(e, SlotSet)]
            assert len(slot_events) >= 1
            user_name_slots = [e for e in slot_events if e.key == "user_name"]
            assert any(e.value == "Jane" for e in user_name_slots)

    @pytest.mark.asyncio
    async def test_send_message_passes_metadata_to_llm(
        self, mcp_task_agent: MCPTaskAgent, mock_output_channel
    ):
        """Test that send_message passes metadata to llm_client.acompletion."""
        # Create agent input with specific metadata
        agent_input = AgentInput(
            id="test_id",
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
            metadata={
                AGENT_METADATA_SENDER_ID_KEY: "user123",
                AGENT_METADATA_AGENT_ID_KEY: "assistant456",
                AGENT_METADATA_MODEL_ID_KEY: "model789",
            },
            timestamp="2024-01-15T10:30:00Z",
        )
        agent_input.recipient_id = "test_user"

        # Create a simple LLM response that will complete immediately
        mock_llm_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Test response"],
            tool_calls=None,
        )

        with (
            patch.object(mcp_task_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
        ):
            mock_llm_client.acompletion = AsyncMock(return_value=mock_llm_response)
            mock_get_tools.return_value = []

            # Call send_message
            await mcp_task_agent.send_message(
                agent_input, output_channel=mock_output_channel
            )

            # Verify acompletion was called
            assert mock_llm_client.acompletion.called

            # Get the expected metadata
            expected_metadata = mcp_task_agent.get_llm_tracing_metadata(agent_input)

            # Verify the metadata parameter was passed correctly
            call_args = mock_llm_client.acompletion.call_args
            assert call_args is not None
            assert "metadata" in call_args.kwargs
            assert call_args.kwargs["metadata"] == expected_metadata

    @pytest.mark.asyncio
    async def test_send_message_calls_process_tool_output_each_iteration(
        self, mcp_task_agent, mock_agent_input, mock_output_channel
    ):
        """process_tool_output is called after each LLM iteration."""
        mock_agent_input.recipient_id = "test_user"
        mock_tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed"],
            tool_calls=[mock_tool_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Done!"],
            tool_calls=[],
        )
        tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )
        seen_tool_result_sizes = []

        async def _capture_process_tool_output(
            current_iteration_tool_results, cumulative_tool_results, output_channel
        ):
            seen_tool_result_sizes.append(
                (
                    len(current_iteration_tool_results),
                    len(cumulative_tool_results),
                )
            )
            return []

        with (
            patch.object(mcp_task_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_task_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_task_agent,
                "process_tool_output",
                new=AsyncMock(side_effect=_capture_process_tool_output),
            ),
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_tool = MagicMock()
            mock_tool.name = "other_tool"
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = tool_output

            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "INPUT_REQUIRED"
        assert seen_tool_result_sizes == [(1, 1)]

    @pytest.mark.asyncio
    async def test_send_message_includes_process_tool_output_events(
        self, mcp_task_agent, mock_agent_input, mock_output_channel
    ):
        """Events from process_tool_output are attached to AgentOutput."""
        mock_agent_input.recipient_id = "test_user"
        mock_tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed"],
            tool_calls=[mock_tool_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Test response"],
            tool_calls=[],
        )
        tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )
        process_events = [BotUttered(text="processed")]

        with (
            patch.object(mcp_task_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_task_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_task_agent,
                "process_tool_output",
                new=AsyncMock(return_value=process_events),
            ),
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_tool = MagicMock()
            mock_tool.name = "other_tool"
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = tool_output

            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "INPUT_REQUIRED"
        assert result.events is not None
        assert any(
            isinstance(event, BotUttered) and event.text == "processed"
            for event in result.events
        )

    @pytest.mark.asyncio
    async def test_send_message_exposes_processed_events_to_next_llm_iteration(
        self, mcp_task_agent, mock_agent_input, mock_output_channel
    ):
        """Processed tool events are added to subsequent LLM iteration context."""
        mock_agent_input.recipient_id = "test_user"
        mock_tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed"],
            tool_calls=[mock_tool_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Final answer"],
            tool_calls=[],
        )
        tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )
        process_events = [BotUttered(text="processed context")]

        with (
            patch.object(mcp_task_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_task_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_task_agent,
                "process_tool_output",
                new=AsyncMock(return_value=process_events),
            ),
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_tool = MagicMock()
            mock_tool.name = "other_tool"
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = tool_output

            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "INPUT_REQUIRED"
        second_call_messages = mock_llm_client.acompletion.call_args_list[1].args[0]
        assert any(
            message.get("role") == "assistant"
            and message.get("content") == "processed context"
            for message in second_call_messages
        )
        assert any(
            isinstance(event, BotUttered) and event.text == "processed context"
            for event in mock_agent_input.events
        )

    @pytest.mark.asyncio
    async def test_send_message_rebuilds_context_and_applies_slot_set_events(
        self, mcp_task_agent, mock_agent_input, mock_output_channel
    ):
        """Rebuilds per-iteration context and applies SlotSet updates to slots."""
        mock_agent_input.recipient_id = "test_user"
        mcp_task_agent._include_date_time = False
        mock_tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed"],
            tool_calls=[mock_tool_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Final answer"],
            tool_calls=[],
        )
        tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )
        process_events = [SlotSet("user_name", "Alice")]
        original_build_messages = mcp_task_agent.build_messages_for_llm_request
        seen_user_names = []

        def _capture_build_messages(context, turns=10):
            slot_values = {slot.name: slot.value for slot in context.slots}
            seen_user_names.append(slot_values.get("user_name"))
            return original_build_messages(context, turns)

        with (
            patch.object(mcp_task_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_task_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_task_agent,
                "process_tool_output",
                new=AsyncMock(return_value=process_events),
            ),
            patch.object(
                mcp_task_agent,
                "build_messages_for_llm_request",
                side_effect=_capture_build_messages,
            ) as mock_build_messages,
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_tool = MagicMock()
            mock_tool.name = "other_tool"
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = tool_output

            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "INPUT_REQUIRED"
        assert mock_build_messages.call_count == 2
        assert seen_user_names == ["John", "Alice"]
        assert any(
            isinstance(event, SlotSet)
            and event.key == "user_name"
            and event.value == "Alice"
            for event in mock_agent_input.events
        )

    @pytest.mark.asyncio
    async def test_send_message_applies_slot_set_for_unknown_slot(
        self, mcp_task_agent, mock_agent_input, mock_output_channel
    ):
        """Unknown SlotSet from process_tool_output is applied in context/output."""
        mock_agent_input.recipient_id = "test_user"
        mock_tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed"],
            tool_calls=[mock_tool_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Final answer"],
            tool_calls=[],
        )
        tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )
        process_events = [SlotSet("unknown_slot", "value")]

        with (
            patch.object(mcp_task_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_task_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_task_agent,
                "process_tool_output",
                new=AsyncMock(return_value=process_events),
            ),
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_tool = MagicMock()
            mock_tool.name = "other_tool"
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = tool_output

            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "INPUT_REQUIRED"
        assert result.events is not None
        assert any(
            isinstance(event, SlotSet) and event.key == "unknown_slot"
            for event in result.events
        )
        assert any(
            isinstance(event, SlotSet) and event.key == "unknown_slot"
            for event in mock_agent_input.events
        )
        assert any(
            slot.name == "unknown_slot" and slot.type == "any"
            for slot in mock_agent_input.slots
        )

    @pytest.mark.asyncio
    async def test_send_message_tool_error_preserves_processed_events(
        self, mcp_task_agent, mock_agent_input, mock_output_channel
    ):
        """Processed tool events survive in output when a tool call fails."""
        mock_agent_input.recipient_id = "test_user"
        ok_call = LLMToolCall(
            id="call_ok",
            type="function",
            tool_name="good_tool",
            tool_args={"arg": "v"},
        )
        bad_call = LLMToolCall(
            id="call_bad",
            type="function",
            tool_name="bad_tool",
            tool_args={"arg": "v"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Running tools"],
            tool_calls=[ok_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["More tools"],
            tool_calls=[bad_call],
        )
        ok_output = AgentToolResult(tool_name="good_tool", result="ok", is_error=False)
        bad_output = AgentToolResult(
            tool_name="bad_tool", result=None, is_error=True, error_message="boom"
        )
        process_events = [BotUttered(text="hook_event")]

        with (
            patch.object(mcp_task_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_task_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_task_agent,
                "process_tool_output",
                new=AsyncMock(return_value=process_events),
            ),
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_good = MagicMock()
            mock_good.name = "good_tool"
            mock_bad = MagicMock()
            mock_bad.name = "bad_tool"
            mock_get_tools.return_value = [mock_good, mock_bad]
            mock_execute_tool.side_effect = [ok_output, bad_output]

            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "FATAL_ERROR"
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_send_message_exit_condition_includes_process_tool_output_events(
        self, mcp_task_agent, mock_agent_input, mock_output_channel
    ):
        """Events from process_tool_output appear in exit-condition completed output.

        A regular tool call populates tool_results so that process_tool_output
        is invoked; on the next iteration the slot-setting tool triggers exit.
        """
        mock_agent_input.recipient_id = "test_user"
        mock_agent_input.metadata = {"exit_if": ["slots.user_name == 'Jane'"]}
        regular_call = LLMToolCall(
            id="call_tool",
            type="function",
            tool_name="lookup_tool",
            tool_args={"q": "Jane"},
        )
        set_slot_call = LLMToolCall(
            id="call_slot",
            type="function",
            tool_name="set_slot_user_name",
            tool_args={"slot_value": "Jane"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Looking up"],
            tool_calls=[regular_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Setting slot"],
            tool_calls=[set_slot_call],
        )
        tool_output = AgentToolResult(
            tool_name="lookup_tool", result="found Jane", is_error=False
        )
        process_events = [BotUttered(text="hook_event")]

        with (
            patch.object(mcp_task_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_task_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_task_agent,
                "process_tool_output",
                new=AsyncMock(return_value=process_events),
            ),
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_lookup = MagicMock()
            mock_lookup.name = "lookup_tool"
            mock_slot = MagicMock()
            mock_slot.name = "set_slot_user_name"
            mock_get_tools.return_value = [mock_lookup, mock_slot]
            mock_execute_tool.return_value = tool_output

            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "COMPLETED"
        assert result.events is not None
        hook_events = [
            e
            for e in result.events
            if isinstance(e, BotUttered) and e.text == "hook_event"
        ]
        slot_events = [e for e in result.events if isinstance(e, SlotSet)]
        assert len(hook_events) == 1
        assert len(slot_events) == 1
        assert slot_events[0].key == "user_name"
        assert slot_events[0].value == "Jane"

    @pytest.mark.asyncio
    async def test_send_message_process_tool_output_failure_returns_fatal_error(
        self, mcp_task_agent, mock_agent_input, mock_output_channel
    ):
        """Failure in process_tool_output should return FATAL_ERROR."""
        mock_agent_input.recipient_id = "test_user"
        mock_tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed"],
            tool_calls=[mock_tool_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Test response"],
            tool_calls=[],
        )
        tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )

        with (
            patch.object(mcp_task_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_task_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_task_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_task_agent,
                "process_tool_output",
                new=AsyncMock(side_effect=Exception("hook failure")),
            ),
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_tool = MagicMock()
            mock_tool.name = "other_tool"
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = tool_output

            result = await mcp_task_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "FATAL_ERROR"
        assert result.error_message is not None
        assert "Failed to process MCP tool output" in result.error_message
