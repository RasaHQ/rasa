import logging
import warnings
import asyncio
from collections import defaultdict
from typing import List, Set, Text
from rasa.core.domain import Domain
from rasa.importers.importer import TrainingDataImporter
from rasa.nlu.training_data import TrainingData
from rasa.core.training.dsl import StoryStep
from rasa.core.training.dsl import UserUttered
from rasa.core.training.dsl import ActionExecuted
from rasa.core.constants import UTTER_PREFIX

logger = logging.getLogger(__name__)


class Validator:
    """A class used to verify usage of intents and utterances."""

    def __init__(self, domain: Domain, intents: TrainingData, stories: List[StoryStep]):
        """Initializes the Validator object. """

        self.domain = domain
        self.intents = intents
        self.stories = stories

    @classmethod
    async def from_importer(cls, importer: TrainingDataImporter) -> "Validator":
        """Create an instance from the domain, nlu and story files."""

        domain = await importer.get_domain()
        stories = await importer.get_stories()
        intents = await importer.get_nlu_data()

        return cls(domain, intents, stories.story_steps)

    def verify_intents(self, ignore_warnings: bool = True) -> bool:
        """Compares list of intents in domain with intents in NLU training data."""

        everything_is_alright = True

        nlu_data_intents = {e.data["intent"] for e in self.intents.intent_examples}

        for intent in self.domain.intents:
            if intent not in nlu_data_intents:
                warnings.warn(
                    f"The intent '{intent}' is listed in the domain file, but "
                    "is not found in the NLU training data."
                )
                everything_is_alright = ignore_warnings and everything_is_alright

        for intent in nlu_data_intents:
            if intent not in self.domain.intents:
                warnings.warn(
                    f"The intent '{intent}' is in the NLU training data, but "
                    f"is not listed in the domain."
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
                intents_string = ", ".join(sorted(intents))
                warnings.warn(
                    f"The example '{text}' was found in these multiples intents: {intents_string }"
                )
        return everything_is_alright

    def verify_intents_in_stories(self, ignore_warnings: bool = True) -> bool:
        """Checks intents used in stories.

        Verifies if the intents used in the stories are valid, and whether
        all valid intents are used in the stories."""

        everything_is_alright = self.verify_intents(ignore_warnings)

        stories_intents = {
            event.intent["name"]
            for story in self.stories
            for event in story.events
            if type(event) == UserUttered
        }

        for story_intent in stories_intents:
            if story_intent not in self.domain.intents:
                warnings.warn(
                    f"The intent '{story_intent}' is used in stories, but is not "
                    f"listed in the domain file."
                )
                everything_is_alright = False

        for intent in self.domain.intents:
            if intent not in stories_intents:
                warnings.warn(f"The intent '{intent}' is not used in any story.")
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
                warnings.warn(
                    f"The utterance '{utterance}' is not listed under 'actions' in the "
                    "domain file. It can only be used as a template."
                )
                everything_is_alright = ignore_warnings and everything_is_alright

        for action in actions:
            if action.startswith(UTTER_PREFIX):
                if action not in utterance_templates:
                    warnings.warn(f"There is no template for utterance '{action}'.")
                    everything_is_alright = False

        return everything_is_alright

    def verify_utterances_in_stories(self, ignore_warnings: bool = True) -> bool:
        """Verifies usage of utterances in stories.

        Checks whether utterances used in the stories are valid,
        and whether all valid utterances are used in stories."""

        everything_is_alright = self.verify_utterances()

        utterance_actions = self._gather_utterance_actions()
        stories_utterances = set()

        for story in self.stories:
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
                    warnings.warn(
                        f"The utterance '{event.action_name}' is used in stories, but is not a "
                        f"valid utterance."
                    )
                    everything_is_alright = False
                stories_utterances.add(event.action_name)

        for utterance in utterance_actions:
            if utterance not in stories_utterances:
                warnings.warn(f"The utterance '{utterance}' is not used in any story.")
                everything_is_alright = ignore_warnings and everything_is_alright

        return everything_is_alright

    def verify_all(self, ignore_warnings: bool = True) -> bool:
        """Runs all the validations on intents and utterances."""

        logger.info("Validating intents...")
        intents_are_valid = self.verify_intents_in_stories(ignore_warnings)

        logger.info("Validating there is no duplications...")
        there_is_no_duplication = self.verify_example_repetition_in_intents(
            ignore_warnings
        )

        logger.info("Validating utterances...")
        stories_are_valid = self.verify_utterances_in_stories(ignore_warnings)
        return intents_are_valid and stories_are_valid and there_is_no_duplication

    def verify_domain_validity(self) -> bool:
        """Checks whether the domain returned by the importer is empty, indicating an invalid domain."""

        return not self.domain.is_empty()
