class BufferedTrackerStore(TrackerStore):
    def __init__(self, flush_interval=5):
        self._pending_events = []
        self.flush_interval = flush_interval

    def update(self, event):
        self._pending_events.append(event)
        if len(self._pending_events) >= self.flush_interval:
            self.flush()

    def flush(self):
        if self._pending_events:
            tracker.update_events(self._pending_events)
            self._pending_events.clear()
            super().save(tracker)
