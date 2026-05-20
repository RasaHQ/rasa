import json
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.shared.core.trackers import DialogueStateTracker
import fast_tracker

class CppTrackerStore(InMemoryTrackerStore):
    def __init__(self, domain, event_broker=None, **kwargs):
        # Trick Rasa by initializing its native memory store to bypass the broken wrapper
        super().__init__(domain, event_broker, **kwargs)
        self.engine = fast_tracker.CppTrackerEngine()
        print("🤖 [SYSTEM INFO] Core Engine initialized using C++ Hijack.")

    async def save(self, tracker, timeout=None):
        serializable = tracker.current_state()
        state_json = json.dumps(serializable)
        print(f"⚡ [C++ WRITE] Routing dialogue state for '{tracker.sender_id}' to C++ layer.")
        self.engine.save(tracker.sender_id, state_json)
        # We do NOT call super().save() because we don't want Python storing the data!

    async def retrieve(self, sender_id):
        state_json = self.engine.retrieve(sender_id)
        if not state_json:
            return None
        print(f"🔍 [C++ READ] Retrieving session state for '{sender_id}' from C++.")
        return DialogueStateTracker.from_dict(
            sender_id, json.loads(state_json), self.domain.slots
        )