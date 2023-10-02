import logging
import re
import string
from collections import defaultdict
from typing import Set, Text, Optional, Dict, Any, List, Tuple

from pypred import Predicate

import rasa.core.training.story_conflict
from rasa.shared.core.flows.flow import (
    ActionFlowStep,
    BranchFlowStep,
    CollectInformationFlowStep,
    FlowsList,
    IfFlowLink,
    SetSlotsFlowStep,
)
import rasa.shared.nlu.constants
from rasa.shared.constants import (
    ASSISTANT_ID_DEFAULT_VALUE,
    ASSISTANT_ID_KEY,
    CONFIG_MANDATORY_KEYS,
    DOCS_URL_DOMAINS,
    DOCS_URL_FORMS,
    UTTER_PREFIX,
    DOCS_URL_ACTIONS,
    REQUIRED_SLOTS_KEY,
)
from rasa.shared.core import constants
from rasa.shared.core.constants import MAPPING_CONDITIONS, ACTIVE_LOOP
from rasa.shared.core.events import ActionExecuted, ActiveLoop
from rasa.shared.core.events import UserUttered
from rasa.shared.core.domain import Domain
from rasa.shared.core.generator import TrainingDataGenerator
from rasa.shared.core.constants import SlotMappingType, MAPPING_TYPE
from rasa.shared.core.slots import ListSlot, Slot
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import TrainingData
import rasa.shared.utils.io

logger = logging.getLogger(__name__)


class Validator:
    """A class used to verify usage of intents and utterances."""

    def __init__(
        self,
        domain: Domain,
        intents: TrainingData,
        story_graph: StoryGraph,
        flows: FlowsList,
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
        self.flows = flows
        self.config = config or {}

    @classmethod
    def from_importer(cls, importer: TrainingDataImporter) -> "Validator":
        """Create an instance from the domain, nlu and story files."""
        domain = importer.get_domain()
        story_graph = importer.get_stories()
        intents = importer.get_nlu_data()
        config = importer.get_config()
        flows = importer.get_flows()

        return cls(domain, intents, story_graph, flows, config)

    def _non_default_intents(self) -> List[Text]:
        return [
            item
            for item in self.domain.intents
            if item not in constants.DEFAULT_INTENTS
        ]

    def verify_intents(self, ignore_warnings: bool = True) -> bool:
        """Compares list of intents in domain with intents in NLU training data."""
        everything_is_alright = True

        nlu_data_intents = {e.data["intent"] for e in self.intents.intent_examples}

        for intent in self._non_default_intents():
            if intent not in nlu_data_intents:
                rasa.shared.utils.io.raise_warning(
                    f"The intent '{intent}' is listed in the domain file, but "
                    f"is not found in the NLU training data."
                )
                everything_is_alright = ignore_warnings or everything_is_alright

        for intent in nlu_data_intents:
            if intent not in self.domain.intents:
                rasa.shared.utils.io.raise_warning(
                    f"There is a message in the training data labeled with intent "
                    f"'{intent}'. This intent is not listed in your domain. You "
                    f"should need to add that intent to your domain file!",
                    docs=DOCS_URL_DOMAINS,
                )
                everything_is_alright = ignore_warnings

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
                everything_is_alright = ignore_warnings
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

        everything_is_alright = self.verify_intents(ignore_warnings=ignore_warnings)

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
                everything_is_alright = ignore_warnings

        for intent in self._non_default_intents():
            if intent not in stories_intents:
                rasa.shared.utils.io.raise_warning(
                    f"The intent '{intent}' is not used in any story or rule."
                )
                everything_is_alright = ignore_warnings or everything_is_alright

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

    def _does_story_only_use_valid_actions(
        self, used_utterances_in_stories: Set[str], utterance_actions: List[str]
    ) -> bool:
        """Checks if all utterances used in stories are valid."""
        has_no_warnings = True
        for used_utterance in used_utterances_in_stories:
            if used_utterance not in utterance_actions:
                rasa.shared.utils.io.raise_warning(
                    f"The action '{used_utterance}' is used in the stories, "
                    f"but is not a valid utterance action. Please make sure "
                    f"the action is listed in your domain and there is a "
                    f"template defined with its name.",
                    docs=DOCS_URL_ACTIONS + "#utterance-actions",
                )
                has_no_warnings = False
        return has_no_warnings

    def _utterances_used_in_stories(self) -> Set[str]:
        """Return all utterances which are used in stories."""
        stories_utterances = set()

        for story in self.story_graph.story_steps:
            for event in story.events:
                if not isinstance(event, ActionExecuted):
                    continue

                if not event.action_name:
                    continue

                if not event.action_name.startswith(UTTER_PREFIX):
                    # we are only interested in utter actions
                    continue

                if event.action_name in stories_utterances:
                    # we already processed this one before, we only want to warn once
                    continue

                stories_utterances.add(event.action_name)
        return stories_utterances

    def _utterances_used_in_flows(self) -> Set[str]:
        """Return all utterances which are used in flows."""
        flow_utterances = set()

        for flow in self.flows.underlying_flows:
            for step in flow.steps:
                if isinstance(step, ActionFlowStep) and step.action.startswith(
                    UTTER_PREFIX
                ):
                    flow_utterances.add(step.action)
                if isinstance(step, CollectInformationFlowStep):
                    flow_utterances.add(step.utter)

                    for rejection in step.rejections:
                        flow_utterances.add(rejection.utter)

        return flow_utterances

    def verify_utterances_in_dialogues(self, ignore_warnings: bool = True) -> bool:
        """Verifies usage of utterances in stories or flows.

        Checks whether utterances used in the stories are valid,
        and whether all valid utterances are used in stories.
        """
        utterance_actions = self._gather_utterance_actions()

        stories_utterances = self._utterances_used_in_stories()
        flow_utterances = self._utterances_used_in_flows()

        all_used_utterances = flow_utterances.union(stories_utterances)

        everything_is_alright = (
            ignore_warnings
            or self._does_story_only_use_valid_actions(
                stories_utterances, list(utterance_actions)
            )
        )

        for utterance in utterance_actions:
            if utterance not in all_used_utterances:
                rasa.shared.utils.io.raise_warning(
                    f"The utterance '{utterance}' is not used in "
                    f"any story, rule or flow."
                )
                everything_is_alright = ignore_warnings or everything_is_alright

        return everything_is_alright

    def verify_forms_in_stories_rules(self) -> bool:
        """Verifies that forms referenced in active_loop directives are present."""
        all_forms_exist = True
        visited_loops = set()

        for story in self.story_graph.story_steps:
            for event in story.events:
                if not isinstance(event, ActiveLoop):
                    continue

                if event.name in visited_loops:
                    # We've seen this loop before, don't alert on it twice
                    continue

                if not event.name:
                    # To support setting `active_loop` to `null`
                    continue

                if event.name not in self.domain.action_names_or_texts:
                    rasa.shared.utils.io.raise_warning(
                        f"The form '{event.name}' is used in the "
                        f"'{story.block_name}' block, but it "
                        f"is not listed in the domain file. You should add it to your "
                        f"domain file!",
                        docs=DOCS_URL_FORMS,
                    )
                    all_forms_exist = False
                visited_loops.add(event.name)

        return all_forms_exist

    def verify_actions_in_stories_rules(self) -> bool:
        """Verifies that actions used in stories and rules are present in the domain."""
        everything_is_alright = True
        visited = set()

        for story in self.story_graph.story_steps:
            for event in story.events:
                if not isinstance(event, ActionExecuted):
                    continue

                if not event.action_name:
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
        stories_are_valid = self.verify_utterances_in_dialogues(ignore_warnings)
        return intents_are_valid and stories_are_valid and there_is_no_duplication

    def verify_form_slots(self) -> bool:
        """Verifies that form slots match the slot mappings in domain."""
        domain_slot_names = [slot.name for slot in self.domain.slots]
        everything_is_alright = True

        for form in self.domain.form_names:
            form_slots = self.domain.required_slots_for_form(form)
            for slot in form_slots:
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

    def verify_slot_mappings(self) -> bool:
        """Verifies that slot mappings match forms."""
        everything_is_alright = True

        for slot in self.domain.slots:
            for mapping in slot.mappings:
                for condition in mapping.get(MAPPING_CONDITIONS, []):
                    condition_active_loop = condition.get(ACTIVE_LOOP)
                    mapping_type = SlotMappingType(mapping.get(MAPPING_TYPE))
                    if (
                        condition_active_loop
                        and condition_active_loop not in self.domain.form_names
                    ):
                        rasa.shared.utils.io.raise_warning(
                            f"Slot '{slot.name}' has a mapping condition for form "
                            f"'{condition_active_loop}' which is not listed in "
                            f"domain forms. Please add this form to the forms section "
                            f"or check for typos."
                        )
                        everything_is_alright = False

                    form_slots = self.domain.forms.get(condition_active_loop, {}).get(
                        REQUIRED_SLOTS_KEY, {}
                    )
                    if (
                        form_slots
                        and slot.name not in form_slots
                        and mapping_type != SlotMappingType.FROM_TRIGGER_INTENT
                    ):
                        rasa.shared.utils.io.raise_warning(
                            f"Slot '{slot.name}' has a mapping condition for form "
                            f"'{condition_active_loop}', but it's not present in "
                            f"'{condition_active_loop}' form's '{REQUIRED_SLOTS_KEY}'. "
                            f"The slot needs to be added to this key."
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

    def warn_if_config_mandatory_keys_are_not_set(self) -> None:
        """Raises a warning if mandatory keys are not present in the config.

        Additionally, raises a UserWarning if the assistant_id key is filled with the
        default placeholder value.
        """
        for key in set(CONFIG_MANDATORY_KEYS):
            if key not in self.config:
                rasa.shared.utils.io.raise_warning(
                    f"The config file is missing the '{key}' mandatory key."
                )

        assistant_id = self.config.get(ASSISTANT_ID_KEY)

        if assistant_id is not None and assistant_id == ASSISTANT_ID_DEFAULT_VALUE:
            rasa.shared.utils.io.raise_warning(
                f"The config file is missing a unique value for the "
                f"'{ASSISTANT_ID_KEY}' mandatory key. Please replace the default "
                f"placeholder value with a unique identifier."
            )

    @staticmethod
    def _log_error_if_slot_not_in_domain(
        slot_name: str,
        domain_slots: Dict[Text, Slot],
        step_id: str,
        flow_id: str,
        all_good: bool,
    ) -> bool:
        if slot_name not in domain_slots:
            logger.error(
                f"The slot '{slot_name}' is used in the "
                f"step '{step_id}' of flow id '{flow_id}', but it "
                f"is not listed in the domain slots. "
                f"You should add it to your domain file!",
            )
            all_good = False

        return all_good

    @staticmethod
    def _log_error_if_list_slot(
        slot: Slot, step_id: str, flow_id: str, all_good: bool
    ) -> bool:
        if isinstance(slot, ListSlot):
            logger.error(
                f"The slot '{slot.name}' is used in the "
                f"step '{step_id}' of flow id '{flow_id}', but it "
                f"is a list slot. List slots are currently not "
                f"supported in flows. You should change it to a "
                f"text, boolean or float slot in your domain file!",
            )
            all_good = False

        return all_good

    @staticmethod
    def _log_error_if_dialogue_stack_slot(
        slot: Slot, step_id: str, flow_id: str, all_good: bool
    ) -> bool:
        if slot.name == constants.DIALOGUE_STACK_SLOT:
            logger.error(
                f"The slot '{constants.DIALOGUE_STACK_SLOT}' is used in the "
                f"step '{step_id}' of flow id '{flow_id}', but it "
                f"is a reserved slot. You must not use reserved slots in "
                f"your flows.",
            )
            all_good = False

        return all_good

    def verify_flows_steps_against_domain(self) -> bool:
        """Checks flows steps' references against the domain file."""
        all_good = True
        domain_slots = {slot.name: slot for slot in self.domain.slots}
        for flow in self.flows.underlying_flows:
            for step in flow.steps:
                if isinstance(step, CollectInformationFlowStep):
                    all_good = self._log_error_if_slot_not_in_domain(
                        step.collect, domain_slots, step.id, flow.id, all_good
                    )
                    current_slot = domain_slots.get(step.collect)
                    if not current_slot:
                        continue

                    all_good = self._log_error_if_list_slot(
                        current_slot, step.id, flow.id, all_good
                    )
                    all_good = self._log_error_if_dialogue_stack_slot(
                        current_slot, step.id, flow.id, all_good
                    )

                elif isinstance(step, SetSlotsFlowStep):
                    for slot in step.slots:
                        slot_name = slot["key"]
                        all_good = self._log_error_if_slot_not_in_domain(
                            slot_name, domain_slots, step.id, flow.id, all_good
                        )
                        current_slot = domain_slots.get(slot_name)
                        if not current_slot:
                            continue

                        all_good = self._log_error_if_list_slot(
                            current_slot, step.id, flow.id, all_good
                        )
                        all_good = self._log_error_if_dialogue_stack_slot(
                            current_slot, step.id, flow.id, all_good
                        )

                elif isinstance(step, ActionFlowStep):
                    regex = r"{context\..+?}"
                    matches = re.findall(regex, step.action)
                    if matches:
                        logger.warning(
                            f"An interpolated action name '{step.action}' was "
                            f"found at step '{step.id}' of flow id '{flow.id}'. "
                            f"Skipping validation for this step. "
                            f"Please make sure that the action name is "
                            f"listed in your domain responses or actions."
                        )
                    elif step.action not in self.domain.action_names_or_texts:
                        logger.error(
                            f"The action '{step.action}' is used in the step "
                            f"'{step.id}' of flow id '{flow.id}', but it "
                            f"is not listed in the domain file. "
                            f"You should add it to your domain file!",
                        )
                        all_good = False
        return all_good

    def verify_unique_flows(self) -> bool:
        """Checks if all flows have unique names and descriptions."""
        all_good = True
        flow_names = set()
        flow_descriptions = set()
        punctuation_table = str.maketrans({i: "" for i in string.punctuation})

        for flow in self.flows.underlying_flows:
            flow_description = flow.description
            cleaned_description = flow_description.translate(punctuation_table)  # type: ignore[union-attr] # noqa: E501
            if cleaned_description in flow_descriptions:
                logger.error(
                    f"Detected duplicate flow description for flow id '{flow.id}'. "
                    f"Flow descriptions must be unique. "
                    f"Please make sure that all flows have different descriptions."
                )
                all_good = False

            if flow.name in flow_names:
                logger.error(
                    f"Detected duplicate flow name '{flow.name}' for flow "
                    f"id '{flow.id}'. Flow names must be unique. "
                    f"Please make sure that all flows have different names."
                )
                all_good = False

            flow_names.add(flow.name)
            flow_descriptions.add(cleaned_description)

        return all_good

    @staticmethod
    def _construct_predicate(
        predicate: Optional[str], step_id: str, all_good: bool = True
    ) -> Tuple[Optional[Predicate], bool]:
        try:
            pred = Predicate(predicate)
        except (TypeError, Exception) as exception:
            logger.error(
                f"Could not initialize the predicate found under step "
                f"'{step_id}': {exception}."
            )
            pred = None
            all_good = False

        return pred, all_good

    def verify_predicates(self) -> bool:
        """Checks that predicates used in branch flow steps or `collect` steps are valid."""  # noqa: E501
        all_good = True
        for flow in self.flows.underlying_flows:
            for step in flow.steps:
                if isinstance(step, BranchFlowStep):
                    for link in step.next.links:
                        if isinstance(link, IfFlowLink):
                            predicate, all_good = Validator._construct_predicate(
                                link.condition, step.id
                            )
                            if predicate and not predicate.is_valid():
                                logger.error(
                                    f"Detected invalid condition '{link.condition}' "
                                    f"at step '{step.id}' for flow id '{flow.id}'. "
                                    f"Please make sure that all conditions are valid."
                                )
                                all_good = False
                elif isinstance(step, CollectInformationFlowStep):
                    predicates = [predicate.if_ for predicate in step.rejections]
                    for predicate in predicates:
                        pred, all_good = Validator._construct_predicate(
                            predicate, step.id
                        )
                        if pred and not pred.is_valid():
                            logger.error(
                                f"Detected invalid rejection '{predicate}' "
                                f"at `collect` step '{step.id}' "
                                f"for flow id '{flow.id}'. "
                                f"Please make sure that all conditions are valid."
                            )
                            all_good = False
        return all_good

    def verify_flows(self) -> bool:
        """Checks for inconsistencies across flows."""
        logger.info("Validating flows...")

        if self.flows.is_empty():
            logger.warning(
                "No flows were found in the data files. "
                "Will not proceed with flow validation.",
            )
            return True

        condition_one = self.verify_flows_steps_against_domain()
        condition_two = self.verify_unique_flows()
        condition_three = self.verify_predicates()

        all_good = all([condition_one, condition_two, condition_three])

        return all_good
