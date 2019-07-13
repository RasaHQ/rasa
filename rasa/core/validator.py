import logging
import asyncio
from typing import List, Text
from rasa.core.domain import Domain
from rasa.nlu.training_data import load_data, TrainingData
from rasa.core.training.dsl import StoryFileReader, StoryStep
from rasa.core.training.dsl import UserUttered
from rasa.core.training.dsl import ActionExecuted
from rasa.core.actions.action import UTTER_PREFIX

logger = logging.getLogger(__name__)


class Validator(object):
    """A class used to verify usage of intents and utterances."""

    def __init__(self, domain: Domain, intents: TrainingData, stories: List[StoryStep]):
        """Initializes the Validator object. """

        self.domain = domain
        self.intents = intents
        self.stories = stories

    @classmethod
    async def from_files(
        cls, domain_file: Text, nlu_data: Text, story_data: Text
    ) -> "Validator":
        """Create an instance from the domain, nlu and story files."""

        domain = Domain.load(domain_file)
        asyncio.new_event_loop()
        stories = await StoryFileReader.read_from_folder(story_data, domain)
        intents = load_data(nlu_data)

        return cls(domain, intents, stories)

    def verify_intents(self):
        """Compares list of intents in domain with intents in NLU training data."""

        domain_intents = set()
        nlu_data_intents = set()

        for intent in self.domain.intent_properties:
            domain_intents.add(intent)

        for intent in self.intents.intent_examples:
            nlu_data_intents.add(intent.data["intent"])

        for intent in domain_intents:
            if intent not in nlu_data_intents:
                logger.warning(
                    "The intent '{}' is listed in the domain file, but "
                    "is not found in the NLU training data.".format(intent)
                )

        for intent in nlu_data_intents:
            if intent not in domain_intents:
                logger.error(
                    "The intent '{}' is in the NLU training data, but "
                    "is not listed in the domain.".format(intent)
                )

        return domain_intents

    def verify_intents_in_stories(self):
        """Checks intents used in stories.

        Verifies if the intents used in the stories are valid, and whether
        all valid intents are used in the stories."""

        domain_intents = self.verify_intents()

        stories_intents = set()
        for story in self.stories:
            for event in story.events:
                if type(event) == UserUttered:
                    intent = event.intent["name"]
                    stories_intents.add(intent)
                    if intent not in domain_intents:
                        logger.error(
                            "The intent '{}' is used in stories, but is not "
                            "listed in the domain file.".format(intent)
                        )

        for intent in domain_intents:
            if intent not in stories_intents:
                logger.warning(
                    "The intent '{}' is not used in any story.".format(intent)
                )

    def verify_utterances(self):
        """Compares list of utterances in actions with utterances in templates."""

        actions = self.domain.action_names
        utterance_templates = set()
        valid_utterances = set()

        for utterance in self.domain.templates:
            utterance_templates.add(utterance)

        for utterance in utterance_templates:
            if utterance in actions:
                valid_utterances.add(utterance)
            else:
                logger.error(
                    "The utterance '{}' is not listed under 'actions' in the "
                    "domain file.".format(utterance)
                )

        for action in actions:
            if action.startswith(UTTER_PREFIX):
                if action not in utterance_templates:
                    logger.error(
                        "There is no template for utterance '{}'.".format(action)
                    )

        return valid_utterances

    def verify_utterances_in_stories(self):
        """Verifies usage of utterances in stories.

        Checks whether utterances used in the stories are valid,
        and whether all valid utterances are used in stories."""

        valid_utterances = self.verify_utterances()
        stories_utterances = set()

        for story in self.stories:
            for event in story.events:
                if isinstance(event, ActionExecuted) and event.action_name.startswith(
                    UTTER_PREFIX
                ):
                    utterance = event.action_name
                    if (
                        utterance not in valid_utterances
                        and utterance not in stories_utterances
                    ):
                        logger.error(
                            "The utterance '{}' is used in stories, but is not a "
                            "valid utterance.".format(utterance)
                        )
                    stories_utterances.add(utterance)

        for utterance in valid_utterances:
            if utterance not in stories_utterances:
                logger.warning(
                    "The utterance '{}' is not used in any story.".format(utterance)
                )

    def verify_all(self):
        """Runs all the validations on intents and utterances."""

        logger.info("Validating intents...")
        self.verify_intents_in_stories()

        logger.info("Validating utterances...")
        self.verify_utterances_in_stories()
