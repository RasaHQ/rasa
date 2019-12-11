from typing import List, Optional, Dict, Text

from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.domain import PREV_PREFIX
from rasa.core.events import Event, ActionExecuted
from rasa.core.featurizers import MaxHistoryTrackerFeaturizer
from rasa.nlu.constants import MESSAGE_INTENT_ATTRIBUTE
from rasa.core.training.generator import TrackerWithCachedStates


class StoryConflict:
    def __init__(
        self,
        sliced_states: List[Optional[Dict[Text, float]]],
        tracker: TrackerWithCachedStates,
        event,
    ):
        self.sliced_states = sliced_states
        self.hash = hash(str(list(sliced_states)))
        self.tracker = (tracker,)
        self.event = event
        self._conflicting_actions = {}  # {"action": ["story_1", ...], ...}
        self.correct_response = None

    @staticmethod
    def find_conflicts(trackers, domain, max_history: int):

        # Create a 'state -> list of actions' dict, where the state is
        # represented by its hash
        rules = {}
        for tracker, event, sliced_states in StoryConflict._sliced_states_stream(
            trackers, domain, max_history
        ):
            h = hash(str(list(sliced_states)))
            if h in rules:
                if event.as_story_string() not in rules[h]:
                    rules[h] += [event.as_story_string()]
            else:
                rules[h] = [event.as_story_string()]

        # Keep only conflicting rules
        rules = {
            state: actions for (state, actions) in rules.items() if len(actions) > 1
        }

        # Iterate once more over all states and note the (unhashed) state,
        # tracker, and event for which a conflict occurs
        conflicts = {}
        for tracker, event, sliced_states in StoryConflict._sliced_states_stream(
            trackers, domain, max_history
        ):
            h = hash(str(list(sliced_states)))
            if h in rules:
                if h not in conflicts:
                    conflicts[h] = StoryConflict(sliced_states, tracker, event)
                conflicts[h].add_conflicting_action(
                    action=event.as_story_string(), story_name=tracker.sender_id
                )

        # Remove conflicts that arise from unpredictable actions
        return [c for (h, c) in conflicts.items() if c.has_prior_events]

    @staticmethod
    def _sliced_states_stream(trackers, domain, max_history):
        for tracker in trackers:
            states = tracker.past_states(domain)
            states = [
                dict(state) for state in states
            ]  # ToDo: Check against rasa/core/featurizers.py:318

            idx = 0
            for event in tracker.events:
                if isinstance(event, ActionExecuted):
                    sliced_states = MaxHistoryTrackerFeaturizer.slice_state_history(
                        states[: idx + 1], max_history
                    )
                    yield tracker, event, sliced_states
                    idx += 1

    def events_prior_to_conflict(self):
        raise NotImplementedError

    @staticmethod
    def _get_prev_event(state) -> [Event, None]:
        if not state:
            return None, None
        result = (None, None)
        for k in state:
            if k.startswith(PREV_PREFIX):
                if k[len(PREV_PREFIX) :] != ACTION_LISTEN_NAME:
                    result = ("action", k[len(PREV_PREFIX) :])
            elif k.startswith(MESSAGE_INTENT_ATTRIBUTE + "_") and not result[0]:
                result = ("intent", k[len(MESSAGE_INTENT_ATTRIBUTE + "_") :])
        return result

    def add_conflicting_action(self, action: Text, story_name: Text):
        if action not in self._conflicting_actions:
            self._conflicting_actions[action] = [story_name]
        else:
            self._conflicting_actions[action] += [story_name]

    @property
    def conflicting_actions(self):
        return list(self._conflicting_actions.keys())

    @property
    def conflicting_actions_with_counts(self):
        return [f"{a} [{len(s)}x]" for (a, s) in self._conflicting_actions.items()]

    @property
    def incorrect_stories(self):
        if self.correct_response:
            incorrect_stories = []
            for stories in [
                s
                for (a, s) in self._conflicting_actions.items()
                if a != self.correct_response
            ]:
                for story in stories:
                    incorrect_stories.append(story)
            return incorrect_stories
        else:
            return []

    @property
    def has_prior_events(self):
        return self._get_prev_event(self.sliced_states[-1])[0] is not None

    def story_prior_to_conflict(self):
        result = ""
        for state in self.sliced_states:
            if state:
                event_type, event_name = self._get_prev_event(state)
                if event_type == "intent":
                    result += f"* {event_name}\n"
                else:
                    result += f"  - {event_name}\n"
        return result

    def __str__(self):
        last_event_type, last_event_name = self._get_prev_event(self.sliced_states[-1])
        if last_event_type:
            conflict_string = f"CONFLICT after {last_event_type} '{last_event_name}':\n"
        else:
            conflict_string = f"CONFLICT at the beginning of stories:\n"
        for action, stories in self._conflicting_actions.items():
            if len(stories) == 1:
                stories = f"'{stories[0]}'"
            elif len(stories) == 2:
                stories = f"'{stories[0]}' and '{stories[1]}'"
            elif len(stories) == 3:
                stories = f"'{stories[0]}', '{stories[1]}', and '{stories[2]}'"
            elif len(stories) >= 4:
                stories = f"'{stories[0]}' and {len(stories) - 1} other trackers"
            conflict_string += f"  {action} predicted in {stories}\n"

        return conflict_string
