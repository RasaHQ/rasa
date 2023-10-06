from rasa.dialogue_understanding.stack.frames.chit_chat_frame import ChitChatStackFrame


def test_chit_chat_frame_type():
    # types should be stable as they are persisted as part of the tracker
    frame = ChitChatStackFrame(frame_id="test")
    assert frame.type() == "chitchat"


def test_chit_chat_frame_from_dict():
    frame = ChitChatStackFrame.from_dict({"frame_id": "test"})
    assert frame.frame_id == "test"
    assert frame.type() == "chitchat"
