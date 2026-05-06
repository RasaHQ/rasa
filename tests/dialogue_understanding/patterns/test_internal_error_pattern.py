from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)


def test_internal_error_pattern_flow_stack_frame_type() -> None:
    frame = InternalErrorPatternFlowStackFrame(flow_id="test")
    assert frame.type() == "pattern_internal_error"


def test_internal_error_pattern_flow_stack_frame_from_dict() -> None:
    frame = InternalErrorPatternFlowStackFrame.from_dict(
        {
            "frame_id": "test_id",
            "step_id": "test_step_id",
            "error_type": "test_error_type",
            "info": {
                "test_info_field": "test_info_value",
                "error_source": "agent",
                "agent_name": "my_agent",
                "agent_type": "mcp_open",
            },
        }
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.flow_id == "pattern_internal_error"
    assert frame.error_type == "test_error_type"
    assert frame.type() == "pattern_internal_error"
    assert frame.info["agent_name"] == "my_agent"
    assert frame.info["agent_type"] == "mcp_open"
    assert frame.info["error_source"] == "agent"


def test_internal_error_context_as_dict_includes_info_for_predicates() -> None:
    frame = InternalErrorPatternFlowStackFrame(
        info={
            "error_source": "mcp_tool",
            "tool_name": "search",
            "mcp_server": "srv",
            "error_message": "failed",
        }
    )
    ctx = frame.context_as_dict([])
    assert ctx["info"] == frame.info
    assert ctx["info"]["error_source"] == "mcp_tool"
    assert ctx["info"]["tool_name"] == "search"
    assert ctx["info"]["mcp_server"] == "srv"
    assert ctx["info"]["error_message"] == "failed"


def test_internal_error_pattern_flow_stack_frame_from_dict_empty_info() -> None:
    frame = InternalErrorPatternFlowStackFrame.from_dict(
        {
            "frame_id": "x",
            "step_id": "y",
            "error_type": "t",
            "info": {},
        }
    )
    assert frame.info == {}
    ctx = frame.context_as_dict([])
    assert "error_source" not in ctx
