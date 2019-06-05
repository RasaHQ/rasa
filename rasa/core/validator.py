import logging
import argparse
import asyncio
from rasa.utils.io import configure_colored_logging
from typing import List, Any, Text
from rasa.core.domain import Domain
from rasa.nlu.training_data import load_data, TrainingData
from rasa.core.training.dsl import StoryFileReader, StoryStep
from rasa.core.training.dsl import UserUttered
from rasa.core.training.dsl import ActionExecuted

logger = logging.getLogger(__name__)


class Validator:
    """Validator is a class to verify the intents and utters written."""

    def __init__(self, domain: Domain, intents: TrainingData, stories: List[StoryStep]):

        """Initialize the validator object. """

        self.domain = domain
        self.intents = intents
        self.valid_intents = []
        self.valid_utterances = []
        self.stories = stories

    def _search(self, vector: List[Any], searched_value: Any):
        """Search for a element in a vector."""
        vector.append(searched_value)
        count = 0
        while searched_value != vector[count]:
            count += 1
        if count == len(vector) - 1:
            return False
        else:
            return True

    def verify_intents(self):
        """Compares list of intents in domain with 
            list of intents in the nlu files."""

        domain_intents = []
        files_intents = []

        for intent in self.domain.intent_properties:
            domain_intents.append(intent)

        for intent in self.intents.intent_examples:
            files_intents.append(intent.data["intent"])

        for intent in domain_intents:
            found = self._search(files_intents, intent)
            if not found:
                logger.error(
                    "The intent '{}' is in the domain file but "
                    "was not found in the intent files".format(intent)
                )
            else:
                self.valid_intents.append(intent)

        for intent in files_intents:
            found = self._search(domain_intents, intent)
            if not found:
                logger.error(
                    "The intent '{}' is in the nlu files but "
                    "was not found in the domain".format(intent)
                )

    def verify_intents_in_stories(self):
        """Verifies if the intents being used in the stories are
            valid and if all the valid intents are being used in
            the stories."""

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
                            "The intent '{}' is used in the "
                            "story files, but it's not a "
                            "valid intent".format(intent)
                        )

        for intent in self.valid_intents:
            found = self._search(stories_intents, intent)
            if not found:
                logger.warning(
                    "The intent '{}' is not being used in any " "story".format(intent)
                )

    def verify_utterances(self):
        """Compares list of utterances in actions with
        list of utterances in the templates."""

        utterance_actions = self.domain.action_names
        utterance_templates = []

        for utterance in self.domain.templates:
            utterance_templates.append(utterance)

        for utterance in utterance_templates:
            found = self._search(utterance_actions, utterance)
            if not found:
                logger.error(
                    "The utterance '{}' is not listed in actions".format(utterance)
                )
            else:
                self.valid_utterances.append(utterance)

        for utterance in utterance_actions:
            if utterance.split("_")[0] == "utter":
                found = self._search(utterance_templates, utterance)
                if not found:
                    logger.error(
                        "There is no template for utterance '{}'".format(utterance)
                    )

    def verify_utterances_in_stories(self):
        """Verifies if the utterances being used in the stories are
        valid and if all the valid utterances are being used in
        the stories."""

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
                            "The utterance '{}' is used in the "
                            "story files, but it's not a "
                            "valid utterance".format(utterance)
                        )

        for utterance in self.valid_utterances:
            found = self._search(stories_utterances, utterance)
            if not found:
                logger.warning(
                    "The utterance '{}' is not being used in any "
                    "story".format(utterance)
                )

    def verify_all(self):
        """Run all the verifications on intents and utterances """

        logger.info("Verifying intents...")
        self.verify_intents_in_stories()

        logger.info("Verifying utterances...")
        self.verify_utterances_in_stories()

    @classmethod
    def from_files(
        cls, domain_file: Text, nlu_data: Text, story_data: Text
    ) -> "Validator":
        """Create an instance from the domain, nlu and story files."""

        domain = Domain.load(domain_file)
        loop = asyncio.get_event_loop()
        stories = loop.run_until_complete(
            StoryFileReader.read_from_folder(story_data, domain)
        )
        intents = load_data(nlu_data)
        return cls(domain, intents, stories)
