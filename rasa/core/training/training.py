from typing import Text, List, TYPE_CHECKING, Dict, Set
from collections import defaultdict

from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.events import SlotSet, ActiveLoop
from rasa.shared.core.constants import SLOTS, ACTIVE_LOOP

if TYPE_CHECKING:
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.trackers import DialogueStateTracker
    from rasa.shared.core.events import Event


def _find_events_after_actions(
    trackers: List["DialogueStateTracker"],
) -> Dict[Text, Set["Event"]]:
    """Creates a dictionary of action names and events that follow these actions.

    Args:
        trackers: the list of trackers

    Returns:
        a dictionary of action names and events that follow these actions
    """
    events_after_actions = defaultdict(set)

    for t in trackers:
        tracker = t.init_copy()
        for event in t.events:
            tracker.update(event)
            if isinstance(event, ActionExecuted):
                continue

            action_name = tracker.latest_action_name
            if action_name:
                events_after_actions[action_name].add(event)

    return events_after_actions


def create_action_fingerprints(
    trackers: List["DialogueStateTracker"], domain: "Domain"
) -> Dict[Text, Dict[Text, List[Text]]]:
    """Fingerprint each action using the events it created during train.

    This allows us to emit warnings when the model is used
    if an action does things it hasn't done during training,
    or if rules are incomplete.

    Args:
        trackers: the list of trackers
        domain: the domain

    Returns:
        a nested dictionary of action names and slots and active loops
            that this action sets
    """
    events_after_actions = _find_events_after_actions(trackers)
    if not events_after_actions:
        return {}

    # take into account only featurized slots
    featurized_slots = {slot.name for slot in domain.slots if slot.has_features()}
    action_fingerprints = defaultdict(dict)
    for action_name, events_after_action in events_after_actions.items():
        slots = list(
            set(
                event.key for event in events_after_action if isinstance(event, SlotSet)
            ).intersection(featurized_slots)
        )
        active_loops = list(
            set(
                event.name
                for event in events_after_action
                if isinstance(event, ActiveLoop)
            )
        )

        if slots:
            action_fingerprints[action_name][SLOTS] = slots
        if active_loops:
            action_fingerprints[action_name][ACTIVE_LOOP] = active_loops

    return action_fingerprints
