from typing import List, Optional, Dict, Text

from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.domain import PREV_PREFIX, Domain
from rasa.core.events import ActionExecuted, Event
from rasa.core.featurizers import MaxHistoryTrackerFeaturizer
from rasa.nlu.constants import MESSAGE_INTENT_ATTRIBUTE
from rasa.core.training.generator import TrackerWithCachedStates


class StoryConflict:
    def __init__(
        self, sliced_states: List[Optional[Dict[Text, float]]],
    ):
        self.sliced_states = sliced_states
        self.hash = hash(str(list(sliced_states)))
        self._conflicting_actions = {}  # {"action": ["story_1", ...], ...}
        self.correct_response = None

    @staticmethod
    def find_conflicts(
        trackers: List[TrackerWithCachedStates], domain: Domain, max_history: int
    ) -> List:
        """
        Generate a list of StoryConflict objects, describing
        conflicts in the given trackers.
        :param trackers: Trackers in which to search for conflicts
        :param domain: The domain
        :param max_history: The maximum history length to be
        taken into account
        :return: List of conflicts
        """

        # We do this in two steps, to reduce memory consumption:

        # Create a 'state -> list of actions' dict, where the state is
        # represented by its hash
        rules = StoryConflict._find_conflicting_states(trackers, domain, max_history)

        # Iterate once more over all states and note the (unhashed) state,
        # for which a conflict occurs
        conflicts = StoryConflict._build_conflicts_from_states(
            trackers, domain, max_history, rules
        )

        return conflicts

    @staticmethod
    def _find_conflicting_states(
        trackers: List[TrackerWithCachedStates], domain: Domain, max_history: int
    ) -> Dict[Text, Optional[List[Text]]]:
        # Create a 'state -> list of actions' dict, where the state is
        # represented by its hash
        rules = {}
        for tracker, event, sliced_states in StoryConflict._sliced_states_iterator(
            trackers, domain, max_history
        ):
            h = hash(str(list(sliced_states)))
            if h not in rules:
                rules[h] = [event.as_story_string()]
            elif h in rules and event.as_story_string() not in rules[h]:
                rules[h] += [event.as_story_string()]

        # Keep only conflicting rules
        return {
            state: actions for (state, actions) in rules.items() if len(actions) > 1
        }

    @staticmethod
    def _build_conflicts_from_states(
        trackers: List["TrackerWithCachedStates"],
        domain: Domain,
        max_history: int,
        rules: Dict[Text, Optional[List[Text]]],
    ):
        # Iterate once more over all states and note the (unhashed) state,
        # for which a conflict occurs
        conflicts = {}
        for tracker, event, sliced_states in StoryConflict._sliced_states_iterator(
            trackers, domain, max_history
        ):
            h = hash(str(list(sliced_states)))

            if h in rules and h not in conflicts:
                conflicts[h] = StoryConflict(sliced_states)

            if h in rules:
                conflicts[h].add_conflicting_action(
                    action=event.as_story_string(), story_name=tracker.sender_id
                )

        # Remove conflicts that arise from unpredictable actions
        return [c for (h, c) in conflicts.items() if c.has_prior_events]

    @staticmethod
    def _sliced_states_iterator(
        trackers: List[TrackerWithCachedStates], domain: Domain, max_history: int
    ) -> (TrackerWithCachedStates, Event, List[Dict[Text, float]]):
        """
        Iterate over all given trackers and all sliced states within
        each tracker, where the slicing is based on `max_history`
        :param trackers: List of trackers
        :param domain: Domain (used for tracker.past_states)
        :param max_history: Assumed `max_history` value for slicing
        :return: Yields (tracker, event, sliced_states) triplet
        """
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

    @staticmethod
    def _get_prev_event(
        state: Optional[Dict[Text, float]]
    ) -> [Optional[Text], Optional[Text]]:
        """
        Returns the type and name of the event (action or intent) previous to the
        given state
        :param state: Element of sliced states
        :return: (type, name) strings of the prior event
        """
        prev_event_type = None
        prev_event_name = None

        for k in state:
            if (
                k.startswith(PREV_PREFIX)
                and k[len(PREV_PREFIX) :] != ACTION_LISTEN_NAME
            ):
                prev_event_type = "action"
                prev_event_name = k[len(PREV_PREFIX) :]

            if not prev_event_type and k.startswith(MESSAGE_INTENT_ATTRIBUTE + "_"):
                prev_event_type = "intent"
                prev_event_name = k[len(MESSAGE_INTENT_ATTRIBUTE + "_") :]

        return prev_event_type, prev_event_name

    def add_conflicting_action(self, action: Text, story_name: Text):
        """
        Add another action that follows from the same state
        :param action: Name of the action
        :param story_name: Name of the story where this action
        is chosen
        """
        if action not in self._conflicting_actions:
            self._conflicting_actions[action] = [story_name]
        else:
            self._conflicting_actions[action] += [story_name]

    @property
    def conflicting_actions(self) -> List[Text]:
        """
        Returns the list of conflicting actions
        """
        return list(self._conflicting_actions.keys())

    @property
    def conflicting_actions_with_counts(self) -> List[Text]:
        """
        Returns a list of strings, describing what action
        occurs how often
        """
        return [f"{a} [{len(s)}x]" for (a, s) in self._conflicting_actions.items()]

    @property
    def incorrect_stories(self) -> List[Text]:
        """
        Returns a list of story names that have not yet been
        corrected.
        """
        if not self.correct_response:
            # Return all stories
            return [v[0] for v in self._conflicting_actions.values()]

        incorrect_stories = []
        story_lists_with_uncorrected_responses = [
            s
            for (a, s) in self._conflicting_actions.items()
            if a != self.correct_response
        ]
        for stories in story_lists_with_uncorrected_responses:
            for story in stories:
                incorrect_stories.append(story)
        return incorrect_stories

    @property
    def has_prior_events(self) -> bool:
        """
        Returns True iff anything has happened before this
        conflict.
        """
        return self._get_prev_event(self.sliced_states[-1])[0] is not None

    def story_prior_to_conflict(self) -> Text:
        """
        Generates a story string, describing the events that
        lead up to the conflict.
        """
        result = ""
        for state in self.sliced_states:
            if not state:
                continue

            event_type, event_name = self._get_prev_event(state)
            if event_type == "intent":
                result += f"* {event_name}\n"
            else:
                result += f"  - {event_name}\n"

        return result

    def __str__(self):
        # Describe where the conflict occurs in the stories
        last_event_type, last_event_name = self._get_prev_event(self.sliced_states[-1])
        if last_event_type:
            conflict_string = f"CONFLICT after {last_event_type} '{last_event_name}':\n"
        else:
            conflict_string = f"CONFLICT at the beginning of stories:\n"

        # List which stories are in conflict with one another
        for action, stories in self._conflicting_actions.items():
            # Summarize if necessary
            story_desc = {
                1: "'{}'",
                2: "'{}' and '{}'",
                3: "'{}', '{}', and '{}'",
            }.get(len(stories))
            if story_desc:
                story_desc = story_desc.format(*stories)
            else:
                # Four or more stories are present
                story_desc = f"'{stories[0]}' and {len(stories) - 1} other trackers"

            conflict_string += f"  {action} predicted in {story_desc}\n"

        return conflict_string
