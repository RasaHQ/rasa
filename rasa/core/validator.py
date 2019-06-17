import logging
import asyncio
from typing import List, Any, Text
from rasa.core.domain import Domain
from rasa.nlu.training_data import load_data, TrainingData
from rasa.core.training.dsl import StoryFileReader, StoryStep
from rasa.core.training.dsl import UserUttered
from rasa.core.training.dsl import ActionExecuted

logger = logging.getLogger(__name__)


class Validator(object):
    """A class used to verify usage of intents and utterances."""

    def __init__(self, domain: Domain, intents: TrainingData, stories: List[StoryStep]):
        """Initializes the Validator object. """

        self.domain = domain
        self.intents = intents
        self.valid_intents = []
        self.valid_utterances = []
        self.stories = stories

    @classmethod
    def from_files(
        cls, domain_file: Text, nlu_data: Text, story_data: Text
    ) -> "Validator":
        """Create an instance from the domain, nlu and story files."""

        domain = Domain.load(domain_file)
        loop = asyncio.new_event_loop()
        stories = loop.run_until_complete(
            StoryFileReader.read_from_folder(story_data, domain)
        )
        intents = load_data(nlu_data)
        return cls(domain, intents, stories)

    def _search(self, vector: List[Any], searched_value: Any) -> bool:
        """Searches for a element in a vector."""

        vector.append(searched_value)
        count = 0
        while searched_value != vector[count]:
            count += 1
        if count == len(vector) - 1:
            return False
        else:
            return True

    def verify_intents(self):
        """Compares list of intents in domain with intents in NLU training data."""

        domain_intents = []
        nlu_data_intents = []

        for intent in self.domain.intent_properties:
            domain_intents.append(intent)

        for intent in self.intents.intent_examples:
            nlu_data_intents.append(intent.data["intent"])

        for intent in domain_intents:
            found = self._search(nlu_data_intents, intent)
            if not found:
                logger.error(
                    "The intent '{}' is listed in the domain file, but "
                    "is not found in the NLU training data.".format(intent)
                )
            else:
                self.valid_intents.append(intent)

        for intent in nlu_data_intents:
            found = self._search(domain_intents, intent)
            if not found:
                logger.error(
                    "The intent '{}' is in the NLU training data, but "
                    "is not listed in the domain.".format(intent)
                )

    def verify_intents_in_stories(self):
        """Checks intents used in stories.

        Verifies if the intents used in the stories are valid, and whether
        all valid intents are used in the stories."""

        if self.valid_intents == []:
            self.verify_intents()

        stories_intents = []
        for story in self.stories:
            for event in story.events:
                if type(event) == UserUttered:
                    intent = event.intent["name"]
                    stories_intents.append(intent)
                    found = self._search(self.valid_intents, intent)

                    if not found:
                        logger.error(
                            "The intent '{}' is used in stories, but is not a "
                            "valid intent.".format(intent)
                        )

        for intent in self.valid_intents:
            found = self._search(stories_intents, intent)
            if not found:
                logger.warning(
                    "The intent '{}' is not used in any story.".format(intent)
                )

    def verify_utterances(self):
        """Compares list of utterances in actions with utterances in templates."""

        actions = self.domain.action_names
        utterance_templates = []

        for utterance in self.domain.templates:
            utterance_templates.append(utterance)

        for utterance in utterance_templates:
            found = self._search(actions, utterance)
            if not found:
                logger.error(
                    "The utterance '{}' is not listed under 'actions' in the domain file.".format(
                        utterance
                    )
                )
            else:
                self.valid_utterances.append(utterance)

        for action in actions:
            if action.split("_")[0] == "utter":
                found = self._search(utterance_templates, action)
                if not found:
                    logger.error(
                        "There is no template for utterance '{}'.".format(utterance)
                    )

    def verify_utterances_in_stories(self):
        """Verifies usage of utterances in stories.

        Checks whether utterances used in the stories are valid,
        and whether all valid utterances are used in stories."""

        if self.valid_utterances == []:
            self.verify_utterances()

        stories_utterances = []

        for story in self.stories:
            for event in story.events:
                if type(event) == ActionExecuted:
                    utterance = event.action_name
                    stories_utterances.append(utterance)
                    found = self._search(self.valid_utterances, utterance)

                    if not found:
                        logger.error(
                            "The utterance '{}' is used in stories, but is not a "
                            "valid utterance.".format(utterance)
                        )

        for utterance in self.valid_utterances:
            found = self._search(stories_utterances, utterance)
            if not found:
                logger.warning(
                    "The utterance '{}' is not used in any "
                    "story.".format(utterance)
                )

    def verify_all(self):
        """Runs all the validations on intents and utterances."""

        logger.info("Validating intents...")
        self.verify_intents_in_stories()

        logger.info("Validating utterances...")
        self.verify_utterances_in_stories()
