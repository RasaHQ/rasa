def test_buffered_tracker_save():
    store = BufferedTrackerStore(flush_interval=2) # example flush interval
    tracker = DialogueStateTracker("user123", slots=[])  # example tracker
    event1 = UserUttered("hi") # example event
    event2 = SlotSet("pizza", "pepperoni") # example event

    store.update(event1)
    assert len(store._pending_events) == 1 # should be 1 after first update

    store.update(event2)
    # flush should trigger
    assert len(store._pending_events) == 0  # should be cleared after flush
