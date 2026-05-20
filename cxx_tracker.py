import json
from rasa.core.tracker_store import TrackerStore
from rasa.shared.core.trackers import DialogueStateTracker
import fast_tracker

class CppTrackerStore(TrackerStore):
    def __init__(self, domain, event_broker=None, **kwargs):
        super().__init__(domain, event_broker, **kwargs)
        self.engine = fast_tracker.CppTrackerEngine()
        print("🤖 [SYSTEM INFO] Core Engine initialized using C++ Hijack.")

    async def save(self, tracker, timeout=None):
        # Extract the serializable list of dictionaries representing tracker events
        events_list = [e.as_dict() for e in tracker.events]
        state_json = json.dumps(events_list)
        
        print(f"⚡ [C++ WRITE] Routing dialogue state for '{tracker.sender_id}' to C++ layer.")
        self.engine.save(tracker.sender_id, state_json)

    async def retrieve(self, sender_id):
        state_json = self.engine.retrieve(sender_id)
        if not state_json:
            return None
            
        print(f"🔍 [C++ READ] Retrieving session state for '{sender_id}' from C++.")
        events_as_dict = json.loads(state_json)
        
        # Reconstruct the DialogueStateTracker by replaying the stored events
        return DialogueStateTracker.from_dict(
            sender_id, 
            events_as_dict, 
            self.domain.slots
        )