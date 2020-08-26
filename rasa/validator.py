import logging
from collections import defaultdict
from typing import Set, Text, Optional, Dict, Any

from packaging import version
from packaging.version import LegacyVersion

import rasa.core.training.story_conflict
from rasa.constants import (
    DOCS_URL_DOMAINS,
    DOCS_URL_ACTIONS,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
    DOCS_BASE_URL,
)
from rasa.core.constants import UTTER_PREFIX
from rasa.core.domain import Domain
from rasa.core.events import ActionExecuted
from rasa.core.events import UserUttered
from rasa.core.training.generator import TrainingDataGenerator
from rasa.core.training.structures import StoryGraph
from rasa.importers.importer import TrainingDataImporter
from rasa.nlu.training_data import TrainingData
from rasa.utils.common import raise_warning

logger = logging.getLogger(__name__)

KEY_TRAINING_DATA_FORMAT_VERSION = "version"


class Validator:
    """A class used to verify usage of intents and utterances."""

    def __init__(
        self, domain: Domain, intents: TrainingData, story_graph: StoryGraph
    ) -> None:
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
                logger.debug(
                    f"The intent '{intent}' is listed in the domain file, but "
                    f"is not found in the NLU training data."
                )
                everything_is_alright = ignore_warnings and everything_is_alright

        for intent in nlu_data_intents:
            if intent not in self.domain.intents:
                raise_warning(
                    f"There is a message in the training data labeled with intent "
                    f"'{intent}'. This intent is not listed in your domain. You "
                    f"should need to add that intent to your domain file!",
                    docs=DOCS_URL_DOMAINS,
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
                raise_warning(
                    f"The example '{text}' was found labeled with multiple "
                    f"different intents in the training data. Each annotated message "
                    f"should only appear with one intent. You should fix that "
                    f"conflict The example is labeled with: {intents_string}."
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
                raise_warning(
                    f"The intent '{story_intent}' is used in your stories, but it "
                    f"is not listed in the domain file. You should add it to your "
                    f"domain file!",
                    docs=DOCS_URL_DOMAINS,
                )
                everything_is_alright = False

        for intent in self.domain.intents:
            if intent not in stories_intents:
                logger.debug(f"The intent '{intent}' is not used in any story.")
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
        """Compares list of utterances in actions with utterances in responses."""

        actions = self.domain.action_names
        utterance_templates = set(self.domain.templates)
        everything_is_alright = True

        for utterance in utterance_templates:
            if utterance not in actions:
                logger.debug(
                    f"The utterance '{utterance}' is not listed under 'actions' in the "
                    f"domain file. It can only be used as a template."
                )
                everything_is_alright = ignore_warnings and everything_is_alright

        for action in actions:
            if action.startswith(UTTER_PREFIX):
                if action not in utterance_templates:
                    raise_warning(
                        f"There is no template for the utterance action '{action}'. "
                        f"The action is listed in your domains action list, but "
                        f"there is no template defined with this name. You should "
                        f"add a template with this key.",
                        docs=DOCS_URL_ACTIONS + "#utterance-actions",
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
                    raise_warning(
                        f"The action '{event.action_name}' is used in the stories, "
                        f"but is not a valid utterance action. Please make sure "
                        f"the action is listed in your domain and there is a "
                        f"template defined with its name.",
                        docs=DOCS_URL_ACTIONS + "#utterance-actions",
                    )
                    everything_is_alright = False
                stories_utterances.add(event.action_name)

        for utterance in utterance_actions:
            if utterance not in stories_utterances:
                logger.debug(f"The utterance '{utterance}' is not used in any story.")
                everything_is_alright = ignore_warnings and everything_is_alright

        return everything_is_alright

    def verify_story_structure(
        self, ignore_warnings: bool = True, max_history: Optional[int] = None
    ) -> bool:
        """Verifies that the bot behaviour in stories is deterministic.

        Args:
            ignore_warnings: When `True`, return `True` even if conflicts were found.
            max_history: Maximal number of events to take into account for conflict identification.

        Returns:
            `False` is a conflict was found and `ignore_warnings` is `False`.
            `True` otherwise.
        """

        logger.info("Story structure validation...")

        trackers = TrainingDataGenerator(
            self.story_graph,
            domain=self.domain,
            remove_duplicates=False,
            augmentation_factor=0,
        ).generate()

        # Create a list of `StoryConflict` objects
        conflicts = rasa.core.training.story_conflict.find_story_conflicts(
            trackers, self.domain, max_history
        )

        if not conflicts:
            logger.info("No story structure conflicts found.")
        else:
            for conflict in conflicts:
                logger.warning(conflict)

        return ignore_warnings or not conflicts

    def verify_nlu(self, ignore_warnings: bool = True) -> bool:
        """Runs all the validations on intents and utterances."""

        logger.info("Validating intents...")
        intents_are_valid = self.verify_intents_in_stories(ignore_warnings)

        logger.info("Validating uniqueness of intents and stories...")
        there_is_no_duplication = self.verify_example_repetition_in_intents(
            ignore_warnings
        )

        logger.info("Validating utterances...")
        stories_are_valid = self.verify_utterances_in_stories(ignore_warnings)
        return intents_are_valid and stories_are_valid and there_is_no_duplication

    def verify_domain_validity(self) -> bool:
        """Checks whether the domain returned by the importer is empty.

        An empty domain is invalid."""

        return not self.domain.is_empty()

    @staticmethod
    def validate_training_data_format_version(
        yaml_file_content: Dict[Text, Any], filename: Text
    ) -> bool:
        """Validates version on the training data content using `version` field
           and warns users if the file is not compatible with the current version of
           Rasa Open Source.

        Args:
            yaml_file_content: Raw content of training data file as a dictionary.
            filename: Name of the validated file.

        Returns:
            `True` if the file can be processed by current version of Rasa Open Source,
            `False` otherwise.
        """
        if not isinstance(yaml_file_content, dict):
            raise ValueError(f"Failed to validate {filename}.")

        version_value = yaml_file_content.get(KEY_TRAINING_DATA_FORMAT_VERSION)

        if not version_value:
            # not raising here since it's not critical
            logger.warning(
                f"Training data file {filename} doesn't have a "
                f"'{KEY_TRAINING_DATA_FORMAT_VERSION}' key. "
                f"Rasa Open Source will read the file as a "
                f"version '{LATEST_TRAINING_DATA_FORMAT_VERSION}' file. "
                f"See {DOCS_BASE_URL}."
            )
            return True

        try:
            parsed_version = version.parse(version_value)
            if isinstance(parsed_version, LegacyVersion):
                raise TypeError

            if version.parse(LATEST_TRAINING_DATA_FORMAT_VERSION) >= parsed_version:
                return True

        except TypeError:
            raise_warning(
                f"Training data file {filename} must specify "
                f"'{KEY_TRAINING_DATA_FORMAT_VERSION}' as string, for example:\n"
                f"{KEY_TRAINING_DATA_FORMAT_VERSION}: '{LATEST_TRAINING_DATA_FORMAT_VERSION}'\n"
                f"Rasa Open Source will read the file as a "
                f"version '{LATEST_TRAINING_DATA_FORMAT_VERSION}' file.",
                docs=DOCS_BASE_URL,
            )
            return True

        raise_warning(
            f"Training data file {filename} has a greater format version than "
            f"your Rasa Open Source installation: "
            f"{version_value} > {LATEST_TRAINING_DATA_FORMAT_VERSION}. "
            f"Please consider updating to the latest version of Rasa Open Source."
            f"This file will be skipped.",
            docs=DOCS_BASE_URL,
        )
        return False
