import logging
from collections import defaultdict
from typing import Set, Text
import questionary
from rasa.core.domain import Domain
from rasa.core.training.generator import TrainingDataGenerator
from rasa.importers.importer import TrainingDataImporter
from rasa.nlu.training_data import TrainingData
from rasa.core.training.structures import StoryGraph
from rasa.core.featurizers import MaxHistoryTrackerFeaturizer
from rasa.core.training.dsl import UserUttered
from rasa.core.training.dsl import ActionExecuted
from rasa.core.constants import UTTER_PREFIX
from rasa.core.story_conflict import StoryConflict

logger = logging.getLogger(__name__)


class Validator:
    """A class used to verify usage of intents and utterances."""

    def __init__(self, domain: Domain, intents: TrainingData, story_graph: StoryGraph):
        """Initializes the Validator object. """

        self.domain = domain
        self.intents = intents
        self.story_graph = story_graph

    @classmethod
    async def from_importer(cls, importer: TrainingDataImporter) -> "Validator":
        """Create an instance from the domain, nlu and story files."""

        domain = await importer.get_domain()
        story_graph = await importer.get_stories()
        intents = await importer.get_nlu_data()

        return cls(domain, intents, story_graph)

    def verify_intents(self, ignore_warnings: bool = True) -> bool:
        """Compares list of intents in domain with intents in NLU training data."""

        everything_is_alright = True

        nlu_data_intents = {e.data["intent"] for e in self.intents.intent_examples}

        for intent in self.domain.intents:
            if intent not in nlu_data_intents:
                logger.warning(
                    "The intent '{}' is listed in the domain file, but "
                    "is not found in the NLU training data.".format(intent)
                )
                everything_is_alright = ignore_warnings and everything_is_alright

        for intent in nlu_data_intents:
            if intent not in self.domain.intents:
                logger.error(
                    "The intent '{}' is in the NLU training data, but "
                    "is not listed in the domain.".format(intent)
                )
                everything_is_alright = False

        return everything_is_alright

    def verify_example_repetition_in_intents(
        self, ignore_warnings: bool = True
    ) -> bool:
        """Checks if there is no duplicated example in different intents."""

        everything_is_alright = True

        duplication_hash = defaultdict(set)
        for example in self.intents.intent_examples:
            text = example.text
            duplication_hash[text].add(example.get("intent"))

        for text, intents in duplication_hash.items():

            if len(duplication_hash[text]) > 1:
                everything_is_alright = ignore_warnings and everything_is_alright
                logger.warning(
                    "The example '{}' was found in these multiples intents: {}".format(
                        text, ", ".join(sorted(intents))
                    )
                )
        return everything_is_alright

    def verify_intents_in_stories(self, ignore_warnings: bool = True) -> bool:
        """Checks intents used in stories.

        Verifies if the intents used in the stories are valid, and whether
        all valid intents are used in the stories."""

        everything_is_alright = self.verify_intents(ignore_warnings)

        stories_intents = {
            event.intent["name"]
            for story in self.story_graph.story_steps
            for event in story.events
            if type(event) == UserUttered
        }

        for story_intent in stories_intents:
            if story_intent not in self.domain.intents:
                logger.error(
                    "The intent '{}' is used in stories, but is not "
                    "listed in the domain file.".format(story_intent)
                )
                everything_is_alright = False

        for intent in self.domain.intents:
            if intent not in stories_intents:
                logger.warning(f"The intent '{intent}' is not used in any story.")
                everything_is_alright = ignore_warnings and everything_is_alright

        return everything_is_alright

    def _gather_utterance_actions(self) -> Set[Text]:
        """Return all utterances which are actions."""
        return {
            utterance
            for utterance in self.domain.templates.keys()
            if utterance in self.domain.action_names
        }

    def verify_utterances(self, ignore_warnings: bool = True) -> bool:
        """Compares list of utterances in actions with utterances in templates."""

        actions = self.domain.action_names
        utterance_templates = set(self.domain.templates)
        everything_is_alright = True

        for utterance in utterance_templates:
            if utterance not in actions:
                logger.warning(
                    "The utterance '{}' is not listed under 'actions' in the "
                    "domain file. It can only be used as a template.".format(utterance)
                )
                everything_is_alright = ignore_warnings and everything_is_alright

        for action in actions:
            if action.startswith(UTTER_PREFIX):
                if action not in utterance_templates:
                    logger.error(f"There is no template for utterance '{action}'.")
                    everything_is_alright = False

        return everything_is_alright

    def verify_utterances_in_stories(self, ignore_warnings: bool = True) -> bool:
        """Verifies usage of utterances in stories.

        Checks whether utterances used in the stories are valid,
        and whether all valid utterances are used in stories."""

        everything_is_alright = self.verify_utterances()

        utterance_actions = self._gather_utterance_actions()
        stories_utterances = set()

        for story in self.story_graph.story_steps:
            for event in story.events:
                if not isinstance(event, ActionExecuted):
                    continue
                if not event.action_name.startswith(UTTER_PREFIX):
                    # we are only interested in utter actions
                    continue

                if event.action_name in stories_utterances:
                    # we already processed this one before, we only want to warn once
                    continue

                if event.action_name not in utterance_actions:
                    logger.error(
                        "The utterance '{}' is used in stories, but is not a "
                        "valid utterance.".format(event.action_name)
                    )
                    everything_is_alright = False
                stories_utterances.add(event.action_name)

        for utterance in utterance_actions:
            if utterance not in stories_utterances:
                logger.warning(f"The utterance '{utterance}' is not used in any story.")
                everything_is_alright = ignore_warnings and everything_is_alright

        return everything_is_alright

    def verify_story_names(self, ignore_warnings: bool = True):
        """Verify that story names are unique."""

        # Tally story names, e.g. {"story_1": 3, "story_2": 1, ...}
        name_tally = {}
        print(self.story_graph.as_story_string())
        for step in self.story_graph.story_steps:
            # print(step.block_name)
            if step.block_name in name_tally:
                name_tally[step.block_name] += 1
            else:
                name_tally[step.block_name] = 1

        # Find story names that appear more than once
        # and construct a warning message
        result = True
        message = ""
        for name, count in name_tally.items():
            if count > 1:
                if result:
                    message = f"Found duplicate story names:\n"
                    result = False
                message += f"  '{name}' appears {count}x\n"

        if result:
            logger.info("All story names are unique")
        else:
            logger.error(message)
        return result

    def verify_story_structure(self,
                               ignore_warnings: bool = True,
                               max_history: int = 5,
                               prompt: bool = False) -> bool:
        """Verifies that bot behaviour in stories is deterministic."""

        logger.info("Story structure validation...")
        logger.info(f"Assuming max_history = {max_history}")

        trackers = TrainingDataGenerator(
            self.story_graph,
            domain=self.domain,
            remove_duplicates=False,   # ToDo: Q&A: Why not remove_duplicates=True?
            augmentation_factor=0).generate()
        rules = {}
        for tracker in trackers:
            states = tracker.past_states(self.domain)
            states = [dict(state) for state in states]  # ToDo: Check against rasa/core/featurizers.py:318

            idx = 0
            for event in tracker.events:
                if isinstance(event, ActionExecuted):
                    sliced_states = MaxHistoryTrackerFeaturizer.slice_state_history(
                        states[: idx + 1], max_history
                    )
                    h = hash(str(list(sliced_states)))
                    if h in rules:
                        if event.as_story_string() not in rules[h]:
                            rules[h] += [event.as_story_string()]
                    else:
                        rules[h] = [event.as_story_string()]
                    idx += 1

        # Keep only conflicting rules
        rules = {state: actions for (state, actions) in rules.items() if len(actions) > 1}

        conflicts = {}

        for tracker in trackers:
            states = tracker.past_states(self.domain)
            states = [dict(state) for state in states]  # ToDo: Check against rasa/core/featurizers.py:318

            idx = 0
            for event in tracker.events:
                if isinstance(event, ActionExecuted):
                    sliced_states = MaxHistoryTrackerFeaturizer.slice_state_history(
                        states[: idx + 1], max_history
                    )
                    h = hash(str(list(sliced_states)))
                    if h in rules:
                        if h not in conflicts:
                            conflicts[h] = StoryConflict(sliced_states, tracker, event)
                        conflicts[h].add_conflicting_action(
                            action=event.as_story_string(),
                            story_name=tracker.sender_id
                        )
                    idx += 1

        # Remove conflicts that arise from unpredictable actions
        conflicts = {h: c for (h, c) in conflicts.items() if c.has_prior_events}

        if len(conflicts) == 0:
            logger.info("No story structure conflicts found.")
        else:
            for conflict in list(conflicts.values()):
                logger.warning(conflict)

                # Fix the conflict if required
                if prompt:
                    print("A conflict occurs after the following sequence of events:")
                    print(conflict.story_prior_to_conflict())
                    keep = "KEEP AS IS"
                    correct_response = questionary.select(
                        message="How should your bot respond at this point?",
                        choices=[keep] + conflict.conflicting_actions_with_counts
                    ).ask()
                    if correct_response != keep:
                        # Remove the story count ending, e.g. " [42x]"
                        conflict.correct_response = correct_response.rsplit(" ", 1)[0]

        for conflict in list(conflicts.values()):
            if conflict.correct_response:
                print(f"Fixing {conflict.incorrect_stories} with {conflict.correct_response}...")

        return len(conflicts) == 0

    def verify_all(self, ignore_warnings: bool = True) -> bool:
        """Runs all the validations on intents and utterances."""

        logger.info("Validating intents...")
        intents_are_valid = self.verify_intents_in_stories(ignore_warnings)

        logger.info("Validating uniqueness of intents and stories...")
        there_is_no_duplication = self.verify_example_repetition_in_intents(
            ignore_warnings
        )
        all_story_names_unique = self.verify_story_names(ignore_warnings)

        logger.info("Validating utterances...")
        stories_are_valid = self.verify_utterances_in_stories(ignore_warnings)
        return (intents_are_valid and stories_are_valid and
                there_is_no_duplication and all_story_names_unique)

    def verify_domain_validity(self) -> bool:
        """Checks whether the domain returned by the importer is empty, indicating an invalid domain."""

        return not self.domain.is_empty()
