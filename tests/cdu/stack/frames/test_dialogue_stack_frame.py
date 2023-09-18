from dataclasses import dataclass
from typing import Any, Dict

import pytest
from rasa.dialogue_understanding.stack.frames.dialogue_stack_frame import (
    DialogueStackFrame,
    InvalidStackFrameType,
    generate_stack_frame_id,
)


@dataclass
class MockStackFrame(DialogueStackFrame):
    @classmethod
    def type(cls) -> str:
        return "mock"


@dataclass
class MockStackFrameWithAdditionalProperty(DialogueStackFrame):

    foo: str = ""

    @classmethod
    def type(cls) -> str:
        return "mock_with_additional_property"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> DialogueStackFrame:
        return MockStackFrameWithAdditionalProperty(
            frame_id=data["frame_id"], foo=data["foo"]
        )


def test_generate_stack_frame_id_generates_different_ids():
    assert generate_stack_frame_id() != generate_stack_frame_id()


def test_dialogue_stack_frame_as_dict():
    frame = MockStackFrame(frame_id="test")

    assert frame.as_dict() == {"frame_id": "test", "type": "mock"}


def test_dialogue_stack_frame_as_dict_contains_additional_attributes():
    frame = MockStackFrameWithAdditionalProperty(foo="foofoo", frame_id="test")

    assert frame.as_dict() == {
        "frame_id": "test",
        "type": "mock_with_additional_property",
        "foo": "foofoo",
    }


def test_dialogue_stack_frame_context_as_dict():
    frame = MockStackFrameWithAdditionalProperty(foo="foofoo", frame_id="test")

    assert frame.context_as_dict([]) == {
        "frame_id": "test",
        "type": "mock_with_additional_property",
        "foo": "foofoo",
    }


def test_create_typed_frame():
    frame = MockStackFrameWithAdditionalProperty(foo="foofoo", frame_id="test")

    assert DialogueStackFrame.create_typed_frame(frame.as_dict()) == frame


def test_create_typed_frame_with_unknown_type():

    with pytest.raises(InvalidStackFrameType):
        DialogueStackFrame.create_typed_frame({"type": "unknown"})
