from rasa.dialogue_understanding.stack.frames.search_frame import SearchStackFrame


def test_search_frame_type():
    # types should be stable as they are persisted as part of the tracker
    frame = SearchStackFrame(frame_id="test")
    assert frame.type() == "search"


def test_search_frame_from_dict():
    frame = SearchStackFrame.from_dict({"frame_id": "test"})
    assert frame.frame_id == "test"
    assert frame.type() == "search"
