import logging
from collections import defaultdict
from typing import Set, Text
from rasa.core.domain import Domain, PREV_PREFIX
from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.training.generator import TrainingDataGenerator
from rasa.importers.importer import TrainingDataImporter
from rasa.nlu.training_data import TrainingData
from rasa.core.training.structures import StoryGraph
from rasa.core.featurizers import MaxHistoryTrackerFeaturizer
from rasa.core.training.dsl import UserUttered
from rasa.core.training.dsl import ActionExecuted
from rasa.core.constants import UTTER_PREFIX

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
                logger.warning(
                    "The intent '{}' is not used in any story.".format(intent)
                )
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
                    logger.error(
                        "There is no template for utterance '{}'.".format(action)
                    )
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
                logger.warning(
                    "The utterance '{}' is not used in any story.".format(utterance)
                )
                everything_is_alright = ignore_warnings and everything_is_alright

        return everything_is_alright

    def verify_story_names(self, ignore_warnings: bool = True):
        """Verify that story names are unique."""
        names = set()
        for step in self.story_graph.story_steps:
            if step.block_name in names:
                logger.warning("Found duplicate story names")
                return ignore_warnings
            names.add(step.block_name)
        logger.info("All story names are unique")
        return True

    def verify_story_structure(self, ignore_warnings: bool = True, max_history: int = 5) -> bool:
        """Verifies that bot behaviour in stories is deterministic."""

        logger.info("Story structure validation...")
        logger.info(f"Assuming max_history = {max_history}")

        trackers = TrainingDataGenerator(
            self.story_graph,
            domain=self.domain,
            remove_duplicates=False,   # ToDo: Q&A: Why don't we deduplicate the graph here?
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
                        known_actions = [info["action"] for info in rules[h]]
                        if event.as_story_string() not in known_actions:
                            # print(f"{h} >> {event.as_story_string()} and {known_actions}")
                            rules[h] += [{
                                "tracker": tracker,
                                "action": event.as_story_string()
                            }]
                    else:
                        # print(f"{h} >> {event.as_story_string()}")
                        rules[h] = [{
                            "tracker": tracker,
                            "action": event.as_story_string()
                        }]
                    idx += 1
        result = True
        for state_hash, info in rules.items():
            if len(info) > 1:
                result = False
                tracker = info[0]["tracker"]
                states = tracker.past_states(self.domain)
                states = [dict(state) for state in states]
                last_event_string = None
                idx = 0
                for event in tracker.events:
                    if isinstance(event, ActionExecuted):
                        sliced_states = MaxHistoryTrackerFeaturizer.slice_state_history(
                            states[: idx + 1], max_history
                        )
                        h = hash(str(list(sliced_states)))
                        if h == state_hash:
                            for k, v in sliced_states[-1].items():
                                if k.startswith(PREV_PREFIX):
                                    if k[len(PREV_PREFIX):] != ACTION_LISTEN_NAME:
                                        last_event_string = f"action '{k[len(PREV_PREFIX):]}'"
                                elif k.startswith("intent_") and not last_event_string:
                                    last_event_string = f"intent '{k[len('intent_'):]}'"
                            break
                        idx += 1
                conflict_string = f"CONFLICT after {last_event_string}:\n"
                for i in info:
                    conflict_string += f"  '{i['action']}' predicted in '{i['tracker'].sender_id}'\n"
                logger.warning(conflict_string)

        if result:
            logger.info("No story structure conflicts found.")

        return result

    def verify_all(self, ignore_warnings: bool = True) -> bool:
        """Runs all the validations on intents and utterances."""

        logger.info("Validating intents...")
        intents_are_valid = self.verify_intents_in_stories(ignore_warnings)

        logger.info("Validating there is no duplications...")
        there_is_no_duplication = self.verify_example_repetition_in_intents(
            ignore_warnings
        )
        all_story_names_unique = self.verify_story_names(ignore_warnings)

        logger.info("Validating utterances...")
        stories_are_valid = self.verify_utterances_in_stories(ignore_warnings)
        return (intents_are_valid and stories_are_valid and
                there_is_no_duplication and all_story_names_unique)
