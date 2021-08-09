import logging
from collections import defaultdict
from typing import Set, Text, Optional, Dict, Any

import rasa.core.training.story_conflict
import rasa.shared.nlu.constants
from rasa.shared.constants import (
    DOCS_URL_DOMAINS,
    UTTER_PREFIX,
    DOCS_URL_ACTIONS,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.events import UserUttered
from rasa.shared.core.generator import TrainingDataGenerator
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.config import RasaNLUModelConfig
import rasa.shared.utils.io

logger = logging.getLogger(__name__)


class Validator:
    """A class used to verify usage of intents and utterances."""

    def __init__(
        self,
        domain: Domain,
        intents: TrainingData,
        story_graph: StoryGraph,
        config: Optional[Dict[Text, Any]],
    ) -> None:
        """Initializes the Validator object.

        Args:
            domain: The domain.
            intents: Training data.
            story_graph: The story graph.
            config: The configuration.
        """
        self.domain = domain
        self.intents = intents
        self.story_graph = story_graph
        self.nlu_config = RasaNLUModelConfig(config)

    @classmethod
    async def from_importer(cls, importer: TrainingDataImporter) -> "Validator":
        """Create an instance from the domain, nlu and story files."""
        domain = await importer.get_domain()
        story_graph = await importer.get_stories()
        intents = await importer.get_nlu_data()
        config = await importer.get_config()

        return cls(domain, intents, story_graph, config)

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
                rasa.shared.utils.io.raise_warning(
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
            text = example.get(rasa.shared.nlu.constants.TEXT)
            duplication_hash[text].add(example.get("intent"))

        for text, intents in duplication_hash.items():

            if len(duplication_hash[text]) > 1:
                everything_is_alright = ignore_warnings and everything_is_alright
                intents_string = ", ".join(sorted(intents))
                rasa.shared.utils.io.raise_warning(
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
            if type(event) == UserUttered and event.intent_name is not None
        }

        for story_intent in stories_intents:
            if story_intent not in self.domain.intents:
                rasa.shared.utils.io.raise_warning(
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
        """Return all utterances which are actions.

        Returns:
            A set of response names found in the domain and data files, with the
            response key stripped in the case of response selector responses.
        """
        domain_responses = {
            response.split(rasa.shared.nlu.constants.RESPONSE_IDENTIFIER_DELIMITER)[0]
            for response in self.domain.responses.keys()
            if response in self.domain.action_names_or_texts
        }
        data_responses = {
            response.split(rasa.shared.nlu.constants.RESPONSE_IDENTIFIER_DELIMITER)[0]
            for response in self.intents.responses.keys()
        }
        return domain_responses.union(data_responses)

    def verify_utterances_in_stories(self, ignore_warnings: bool = True) -> bool:
        """Verifies usage of utterances in stories.

        Checks whether utterances used in the stories are valid,
        and whether all valid utterances are used in stories.
        """
        everything_is_alright = True

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
                    rasa.shared.utils.io.raise_warning(
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

    def verify_actions_in_stories_rules(self) -> bool:
        """Verifies that actions used in stories and rules are present in the domain."""
        everything_is_alright = True
        visited = set()

        for story in self.story_graph.story_steps:
            for event in story.events:
                if not isinstance(event, ActionExecuted):
                    continue

                if not event.action_name.startswith("action_"):
                    continue

                if event.action_name in visited:
                    # we already processed this one before, we only want to warn once
                    continue

                if event.action_name not in self.domain.action_names_or_texts:
                    rasa.shared.utils.io.raise_warning(
                        f"The action '{event.action_name}' is used in the "
                        f"'{story.block_name}' block, but it "
                        f"is not listed in the domain file. You should add it to your "
                        f"domain file!",
                        docs=DOCS_URL_DOMAINS,
                    )
                    everything_is_alright = False
                visited.add(event.action_name)

        return everything_is_alright

    def verify_story_structure(
        self, ignore_warnings: bool = True, max_history: Optional[int] = None
    ) -> bool:
        """Verifies that the bot behaviour in stories is deterministic.

        Args:
            ignore_warnings: When `True`, return `True` even if conflicts were found.
            max_history: Maximal number of events to take into account for conflict
                identification.

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
        ).generate_story_trackers()

        # Create a list of `StoryConflict` objects
        conflicts = rasa.core.training.story_conflict.find_story_conflicts(
            trackers, self.domain, max_history, self.nlu_config
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

    def verify_form_slots(self) -> bool:
        """Verifies that form slots match the slot mappings in domain."""
        domain_slot_names = [slot.name for slot in self.domain.slots]
        everything_is_alright = True

        for form in self.domain.form_names:
            form_slots = self.domain.slot_mapping_for_form(form)
            for slot in form_slots.keys():
                if slot in domain_slot_names:
                    continue
                else:
                    rasa.shared.utils.io.raise_warning(
                        f"The form slot '{slot}' in form '{form}' "
                        f"is not present in the domain slots."
                        f"Please add the correct slot or check for typos.",
                        docs=DOCS_URL_DOMAINS,
                    )
                    everything_is_alright = False

        return everything_is_alright

    def verify_domain_validity(self) -> bool:
        """Checks whether the domain returned by the importer is empty.

        An empty domain or one that uses deprecated Mapping Policy is invalid.
        """
        if self.domain.is_empty():
            return False

        for intent_key, intent_dict in self.domain.intent_properties.items():
            if "triggers" in intent_dict:
                rasa.shared.utils.io.raise_warning(
                    f"The intent {intent_key} in the domain file "
                    f"is using the MappingPolicy format "
                    f"which has now been deprecated. "
                    f"Please migrate to RulePolicy."
                )
                return False

        return True
