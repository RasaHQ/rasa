import logging
import argparse
import asyncio
from rasa import utils
from typing import List, Any, Text
from rasa.core.domain import Domain
from rasa.nlu.training_data import load_data, TrainingData
from rasa.core.training.dsl import StoryFileReader, StoryStep
from rasa.core.training.dsl import UserUttered
from rasa.core.training.dsl import ActionExecuted
from rasa.core import cli

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Parse all the command line arguments for the run script."""

    parser = argparse.ArgumentParser(description="Validates files")

    parser.add_argument(
        "--domain", "-d", type=str, required=True,
        help="Path for the domain file"
    )

    parser.add_argument(
        "--stories",
        "-s",
        type=str,
        required=True,
        help="Path for the stories file or directory",
    )

    parser.add_argument(
        "--intents",
        "-i",
        type=str,
        required=True,
        help="Path for the intents file or directory",
    )

    parser.add_argument(
        "--skip-intents-validation",
        action="store_true",
        default=False,
        help="Skips validations to intents",
    )

    parser.add_argument(
        "--skip-utterances-validation",
        action="store_true",
        default=False,
        help="Skips validations to utterances",
    )

    cli.arguments.add_logging_option_arguments(parser)
    cli.run.add_run_arguments(parser)
    return parser


class Validator:
    def __init__(self,
                 domain: Domain,
                 intents: TrainingData,
                 stories: List[StoryStep]):
        self.domain = domain
        self.intents = intents
        self.valid_intents = []
        self.valid_utterances = []
        self.stories = stories

    def _search(self, vector: List[Any], searched_value: Any):
        vector.append(searched_value)
        count = 0
        while searched_value != vector[count]:
            count += 1
        if count == len(vector) - 1:
            return False
        else:
            return True

    def verify_intents(self):
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
                    "The intent '{}' is not being used in any "
                    "story".format(intent)
                )

    def verify_utterances(self):
        utterance_actions = self.domain.action_names
        utterance_templates = []

        for utterance in self.domain.templates:
            utterance_templates.append(utterance)

        for utterance in utterance_templates:
            found = self._search(utterance_actions, utterance)
            if not found:
                logger.error("The utterance '{}' is not listed in actions"
                             .format(utterance))
            else:
                self.valid_utterances.append(utterance)

        for utterance in utterance_actions:
            if utterance.split("_")[0] == "utter":
                found = self._search(utterance_templates, utterance)
                if not found:
                    logger.error("There is no template for utterance '{}'"
                                 .format(utterance))

    def verify_utterances_in_stories(self):
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
        logger.info("Verifying intents...")
        self.verify_intents_in_stories()

        logger.info("Verifying utterances...")
        self.verify_utterances_in_stories()

    @classmethod
    def from_files(cls,
                   domain_file: Text,
                   nlu_data: Text,
                   story_data: Text) -> 'Validator':
        """Create an instance from the domain, nlu and story files."""

        domain = Domain.load(domain_file)
        stories = asyncio.run(
            StoryFileReader.read_from_folder(story_data, domain)
        )
        intents = load_data(nlu_data)
        return cls(domain, intents, stories)


if __name__ == "__main__":
    parser = create_argument_parser()
    cmdline_args = parser.parse_args()
    utils.configure_colored_logging(cmdline_args.loglevel)

    validator = Validator.from_files(cmdline_args.domain, cmdline_args.intents, cmdline_args.stories)

    skip_intents_validation = cmdline_args.skip_intents_validation
    skip_utterances_validation = cmdline_args.skip_utterances_validation

    validator.verify_all()
