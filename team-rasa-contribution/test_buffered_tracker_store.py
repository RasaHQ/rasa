def test_buffered_tracker_save():
    store = BufferedTrackerStore(flush_interval=2)
    tracker = DialogueStateTracker("user123", slots=[])
    event1 = UserUttered("hi")
    event2 = SlotSet("pizza", "pepperoni")

    store.update(event1)
    assert len(store._pending_events) == 1

    store.update(event2)
    # flush should trigger
    assert len(store._pending_events) == 0  # should be cleared after flush
