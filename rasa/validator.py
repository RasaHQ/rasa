import logging
import structlog
import re
import string
from collections import defaultdict
from typing import Set, Text, Optional, Dict, Any, List, Tuple

from jinja2 import Template
from pypred import Predicate
from pypred.ast import Literal, CompareOperator, NegateOperator

import rasa.core.training.story_conflict
from rasa.core.channels import UserMessage
from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.shared.core.command_payload_reader import (
    CommandPayloadReader,
    MAX_NUMBER_OF_SLOTS,
)
from rasa.shared.core.flows.flow_step_links import IfFlowStepLink
from rasa.shared.core.flows.steps.set_slots import SetSlotsFlowStep
from rasa.shared.core.flows.steps.collect import CollectInformationFlowStep
from rasa.shared.core.flows.steps.action import ActionFlowStep
from rasa.shared.core.flows.steps.link import LinkFlowStep
from rasa.shared.core.flows import FlowsList
import rasa.shared.nlu.constants
from rasa.shared.constants import (
    ASSISTANT_ID_DEFAULT_VALUE,
    ASSISTANT_ID_KEY,
    CONFIG_MANDATORY_KEYS,
    DOCS_URL_DOMAIN,
    DOCS_URL_DOMAINS,
    DOCS_URL_FORMS,
    DOCS_URL_RESPONSES,
    UTTER_PREFIX,
    DOCS_URL_ACTIONS,
    REQUIRED_SLOTS_KEY,
)
from rasa.shared.core import constants
from rasa.shared.core.constants import MAPPING_CONDITIONS, ACTIVE_LOOP
from rasa.shared.core.events import ActionExecuted, ActiveLoop
from rasa.shared.core.events import UserUttered
from rasa.shared.core.domain import (
    Domain,
    RESPONSE_KEYS_TO_INTERPOLATE,
)
from rasa.shared.core.generator import TrainingDataGenerator
from rasa.shared.core.constants import SlotMappingType, MAPPING_TYPE
from rasa.shared.core.slots import BooleanSlot, CategoricalSlot, ListSlot, Slot
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.data import create_regex_pattern_reader
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.constants import COMMANDS
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
import rasa.shared.utils.cli
import rasa.shared.utils.io

logger = logging.getLogger(__name__)
structlogger = structlog.get_logger()


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
            flows: The flows.
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
                structlogger.warn(
                    "validator.verify_intents.not_in_nlu_training_data",
                    intent=intent,
                    event_info=(
                        f"The intent '{intent}' is listed "
                        f"in the domain file, but is not found "
                        f"in the NLU training data."
                    ),
                )
                everything_is_alright = ignore_warnings or everything_is_alright

        for intent in nlu_data_intents:
            if intent not in self.domain.intents:
                structlogger.warn(
                    "validator.verify_intents.not_in_domain",
                    intent=intent,
                    event_info=(
                        f"There is a message in the training data "
                        f"labeled with intent '{intent}'. This "
                        f"intent is not listed in your domain. You "
                        f"should need to add that intent to your domain "
                        f"file!"
                    ),
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
                structlogger.warn(
                    "validator.verify_example_repetition_in_intents"
                    ".one_example_multiple_intents",
                    example=text,
                    intents=intents_string,
                    event_info=(
                        f"The example '{text}' was found labeled "
                        f"with multiple different intents in the "
                        f"training data. Each annotated message "
                        f"should only appear with one intent. "
                        f"You should fix that conflict The example is "
                        f"labeled with: {intents_string}."
                    ),
                )
        return everything_is_alright

    def verify_intents_in_stories_or_flows(self, ignore_warnings: bool = True) -> bool:
        """Checks intents used in stories.

        Verifies if the intents used in the stories are valid, and whether
        all valid intents are used in the stories.
        """
        everything_is_alright = self.verify_intents(ignore_warnings=ignore_warnings)

        stories_intents = {
            event.intent["name"]
            for story in self.story_graph.story_steps
            for event in story.events
            if type(event) == UserUttered and event.intent_name is not None
        }
        flow_intents = {
            trigger.intent
            for flow in self.flows.underlying_flows
            if flow.nlu_triggers is not None
            for trigger in flow.nlu_triggers.trigger_conditions
        }
        used_intents = stories_intents.union(flow_intents)

        for intent in used_intents:
            if intent not in self.domain.intents:
                structlogger.warn(
                    "validator.verify_intents_in_stories_or_flows.not_in_domain",
                    intent=intent,
                    event_info=(
                        f"The intent '{intent}' is used in a "
                        f"story or flow, but it is not listed in "
                        f"the domain file. You should add it to your "
                        f"domain file!"
                    ),
                    docs=DOCS_URL_DOMAINS,
                )
                everything_is_alright = ignore_warnings

        for intent in self._non_default_intents():
            if intent not in used_intents:
                structlogger.warn(
                    "validator.verify_intents_in_stories_or_flows.not_used",
                    intent=intent,
                    event_info=(
                        f"The intent '{intent}' is not used "
                        f"in any story, rule or flow."
                    ),
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
                structlogger.warn(
                    "validator.invalid_utterance_action",
                    action=used_utterance,
                    event_info=(
                        f"The action '{used_utterance}' is used in the stories, "
                        f"but is not a valid utterance action. Please make sure "
                        f"the action is listed in your domain and there is a "
                        f"template defined with its name."
                    ),
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

    @classmethod
    def check_for_placeholder(cls, value: Any) -> bool:
        """Check if a value contains a placeholder."""
        if isinstance(value, str):
            return bool(re.search(r"{\s*}", value))
        elif isinstance(value, dict):
            return any(cls.check_for_placeholder(i) for i in value.values())
        elif isinstance(value, list):
            return any(cls.check_for_placeholder(i) for i in value)
        return False

    def check_for_no_empty_paranthesis_in_responses(self) -> bool:
        """Checks if there are no empty parenthesis in utterances."""
        everything_is_alright = True

        for response_text, response_variations in self.domain.responses.items():
            for response in response_variations:
                if any(
                    self.check_for_placeholder(response.get(key))
                    for key in RESPONSE_KEYS_TO_INTERPOLATE
                ):
                    structlogger.error(
                        "validator.empty_paranthesis_in_utterances",
                        response=response_text,
                        event_info=(
                            f"The response '{response_text}' in the domain file "
                            f"contains empty parenthesis, which is not permitted."
                            f" Please remove the empty parenthesis."
                        ),
                    )
                    everything_is_alright = False

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
                    structlogger.error(
                        "validator.verify_forms_in_stories_rules.not_in_domain",
                        form=event.name,
                        block=story.block_name,
                        event_info=(
                            f"The form '{event.name}' is used in the "
                            f"'{story.block_name}' block, but it "
                            f"is not listed in the domain file. "
                            f"You should add it to your "
                            f"domain file!"
                        ),
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
                    structlogger.error(
                        "validator.verify_actions_in_stories_rules.not_in_domain",
                        action=event.action_name,
                        block=story.block_name,
                        event_info=(
                            f"The action '{event.action_name}' is used in the "
                            f"'{story.block_name}' block, but it "
                            f"is not listed in the domain file. You "
                            f"should add it to your domain file!"
                        ),
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
        structlogger.info(
            "validator.verify_story_structure.start",
            event_info="Story structure validation...",
        )

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
            structlogger.info(
                "validator.verify_story_structure.no_conflicts",
                event_info="No story structure conflicts found.",
            )
        else:
            for conflict in conflicts:
                structlogger.warn(
                    "validator.verify_story_structure.conflicts",
                    event_info="Found story structure conflict",
                    conflict=str(conflict),
                )

        return ignore_warnings or not conflicts

    def verify_nlu(self, ignore_warnings: bool = True) -> bool:
        """Runs all the validations on intents and utterances."""
        structlogger.info(
            "validator.verify_intents_in_stories.start",
            event_info="Validating intents...",
        )
        intents_are_valid = self.verify_intents_in_stories_or_flows(ignore_warnings)

        structlogger.info(
            "validator.verify_example_repetition_in_intents.start",
            event_info="Validating uniqueness of intents and stories...",
        )
        there_is_no_duplication = self.verify_example_repetition_in_intents(
            ignore_warnings
        )

        return intents_are_valid and there_is_no_duplication

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
                    structlogger.warn(
                        "validator.verify_form_slots.not_in_domain",
                        slot=slot,
                        form=form,
                        event_info=(
                            f"The form slot '{slot}' in form '{form}' "
                            f"is not present in the domain slots."
                            f"Please add the correct slot or check for typos."
                        ),
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
                        structlogger.warn(
                            "validator.verify_slot_mappings.not_in_domain",
                            slot=slot.name,
                            form=condition_active_loop,
                            event_info=(
                                f"Slot '{slot.name}' has a mapping "
                                f"condition for form '{condition_active_loop}' "
                                f"which is not listed in domain forms. Please "
                                f"add this form to the forms section or check "
                                f"for typos."
                            ),
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
                        structlogger.warn(
                            "validator.verify_slot_mappings.not_in_forms_key",
                            slot=slot.name,
                            form=condition_active_loop,
                            forms_key=REQUIRED_SLOTS_KEY,
                            event_info=(
                                f"Slot '{slot.name}' has a mapping condition "
                                f"for form '{condition_active_loop}', but it's "
                                f"not present in '{condition_active_loop}' "
                                f"form's '{REQUIRED_SLOTS_KEY}'. "
                                f"The slot needs to be added to this key."
                            ),
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
                structlogger.warn(
                    "validator.verify_domain_validity.mapping_policy_deprecation",
                    intent_key=intent_key,
                    event_info=(
                        f"The intent {intent_key} in the domain file "
                        f"is using the MappingPolicy format "
                        f"which has now been deprecated. "
                        f"Please migrate to RulePolicy."
                    ),
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
                structlogger.warn(
                    "validator.config_missing_mandatory_key",
                    key=key,
                    event_info=(
                        f"The config file is missing the " f"'{key}' mandatory key."
                    ),
                )

        assistant_id = self.config.get(ASSISTANT_ID_KEY)

        if assistant_id is not None and assistant_id == ASSISTANT_ID_DEFAULT_VALUE:
            structlogger.warn(
                "validator.config_missing_unique_mandatory_key_value",
                key=ASSISTANT_ID_KEY,
                event_info=(
                    f"The config file is missing a unique value for the "
                    f"'{ASSISTANT_ID_KEY}' mandatory key. Please replace the default "
                    f"placeholder value with a unique identifier."
                ),
            )

    def _log_error_if_either_action_or_utterance_are_not_defined(
        self,
        collect: CollectInformationFlowStep,
        all_good: bool,
        domain_slots: Dict[Text, Slot],
    ) -> bool:
        """Validates that a collect step can have either an action or an utterance.
        Also logs an error if neither an action nor an utterance is defined.

        Args:
            collect: the name of the slot to collect
            all_good: boolean value indicating the validation status

        Returns:
            False, if validation failed, true, otherwise
        """
        has_utterance_defined = any(
            [u for u in self.domain.utterances_for_response if u == collect.utter]
        )

        has_action_defined = any(
            [
                a
                for a in self.domain.action_names_or_texts
                if a == collect.collect_action
            ]
        )

        if has_utterance_defined and has_action_defined:
            structlogger.error(
                "validator.verify_flows_steps_against_domain.collect_step",
                collect=collect.collect,
                has_utterance_defined=has_utterance_defined,
                has_action_defined=has_action_defined,
                event_info=(
                    f"The collect step '{collect.collect}' has an utterance "
                    f"'{collect.utter}' as well as an action "
                    f"'{collect.collect_action}' defined. "
                    f"You can just have one of them! "
                    f"Please remove either the utterance or the action."
                ),
            )
            all_good = False

        slot = domain_slots.get(collect.collect)
        slot_has_initial_value_defind = slot and slot.initial_value is not None

        if (
            not slot_has_initial_value_defind
            and not has_utterance_defined
            and not has_action_defined
        ):
            structlogger.error(
                "validator.verify_flows_steps_against_domain.collect_step",
                collect=collect.collect,
                has_utterance_defined=has_utterance_defined,
                has_action_defined=has_action_defined,
                event_info=(
                    f"The collect step '{collect.collect}' has neither an utterance "
                    f"nor an action defined, or an initial value defined in the domain."
                    f"You need to define either an utterance or an action."
                ),
            )
            all_good = False

        return all_good

    @staticmethod
    def _log_error_if_slot_not_in_domain(
        slot_name: str,
        domain_slots: Dict[Text, Slot],
        step_id: str,
        flow_id: str,
        all_good: bool,
    ) -> bool:
        if slot_name not in domain_slots:
            structlogger.error(
                "validator.verify_flows_steps_against_domain.slot_not_in_domain",
                slot=slot_name,
                step=step_id,
                flow=flow_id,
                event_info=(
                    f"The slot '{slot_name}' is used in the "
                    f"step '{step_id}' of flow id '{flow_id}', but it "
                    f"is not listed in the domain slots. "
                    f"You should add it to your domain file!"
                ),
            )
            all_good = False

        return all_good

    @staticmethod
    def _log_error_if_list_slot(
        slot: Slot, step_id: str, flow_id: str, all_good: bool
    ) -> bool:
        if isinstance(slot, ListSlot):
            structlogger.error(
                "validator.verify_flows_steps_against_domain.use_of_list_slot_in_flow",
                slot=slot.name,
                step=step_id,
                flow=flow_id,
                event_info=(
                    f"The slot '{slot.name}' is used in the "
                    f"step '{step_id}' of flow id '{flow_id}', but it "
                    f"is a list slot. List slots are currently not "
                    f"supported in flows. You should change it to a "
                    f"text, boolean or float slot in your domain file!"
                ),
            )
            all_good = False

        return all_good

    def verify_flows_steps_against_domain(self) -> bool:
        """Checks flows steps' references against the domain file."""
        all_good = True
        domain_slots = {slot.name: slot for slot in self.domain.slots}
        flow_ids = [flow.id for flow in self.flows.underlying_flows]

        for flow in self.flows.underlying_flows:
            for step in flow.steps_with_calls_resolved:
                if isinstance(step, CollectInformationFlowStep):
                    all_good = (
                        self._log_error_if_either_action_or_utterance_are_not_defined(
                            step, all_good, domain_slots
                        )
                    )

                    all_good = self._log_error_if_slot_not_in_domain(
                        step.collect, domain_slots, step.id, flow.id, all_good
                    )

                    current_slot = domain_slots.get(step.collect)
                    if not current_slot:
                        continue

                    all_good = self._log_error_if_list_slot(
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

                elif isinstance(step, ActionFlowStep):
                    regex = r"{context\..+?}"
                    matches = re.findall(regex, step.action)
                    if matches:
                        structlogger.debug(
                            "validator.verify_flows_steps_against_domain"
                            ".interpolated_action",
                            action=step.action,
                            step=step.id,
                            flow=flow.id,
                            event_info=(
                                f"An interpolated action name '{step.action}' was "
                                f"found at step '{step.id}' of flow id '{flow.id}'. "
                                f"Skipping validation for this step. "
                                f"Please make sure that the action name is "
                                f"listed in your domain responses or actions."
                            ),
                        )
                    elif step.action not in self.domain.action_names_or_texts:
                        structlogger.error(
                            "validator.verify_flows_steps_against_domain"
                            ".action_not_in_domain",
                            action=step.action,
                            step=step.id,
                            flow=flow.id,
                            event_info=(
                                f"The action '{step.action}' is used in the "
                                f"step '{step.id}' of flow id '{flow.id}', "
                                f"but it is not listed in the domain file. "
                                f"You should add it to your domain file!"
                            ),
                        )
                        all_good = False

                elif isinstance(step, LinkFlowStep):
                    if step.link not in flow_ids:
                        logger.error(
                            f"The flow '{step.link}' is used in the step "
                            f"'{step.id}' of flow id '{flow.id}', but it "
                            f"is not listed in the flows file. "
                            f"Did you make a typo?",
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
            cleaned_description = flow_description.translate(punctuation_table)  # type: ignore[union-attr]
            if cleaned_description in flow_descriptions:
                structlogger.error(
                    "validator.verify_unique_flows.duplicate_description",
                    flow=flow.id,
                    event_info=(
                        f"Detected duplicate flow description for "
                        f"flow id '{flow.id}'. Flow descriptions must be "
                        f"unique. Please make sure that all flows have "
                        f"different descriptions."
                    ),
                )
                all_good = False

            if flow.name in flow_names:
                structlogger.error(
                    "validator.verify_unique_flows.duplicate_name",
                    flow=flow.id,
                    name=flow.name,
                    event_info=(
                        f"Detected duplicate flow name '{flow.name}' for flow "
                        f"id '{flow.id}'. Flow names must be unique. "
                        f"Please make sure that all flows have different names."
                    ),
                )
                all_good = False

            flow_names.add(flow.name)
            flow_descriptions.add(cleaned_description)

        return all_good

    def _build_context(self) -> Dict[str, Any]:
        """Build context for jinja template rendering.

        Returns:
            A dictionary containing the allowed namespaces for jinja template rendering:
            - `context`: The context mapping the attributes of every flow stack frame
                to None values because this is only used for rendering the template
                during validation.
            - `slots`: The slots of the domain mapped to None values because this is
                only used for rendering the template during validation and not for
                evaluating the predicate at runtime.
        """
        subclasses = [subclass for subclass in PatternFlowStackFrame.__subclasses__()]
        subclass_attrs = []
        for subclass in subclasses:
            subclass_attrs.extend(
                [attr for attr in dir(subclass) if not attr.startswith("__")]
            )

        context = {
            "context": {attr: None for attr in subclass_attrs},
            "slots": {slot.name: None for slot in self.domain.slots},
        }
        return context

    @staticmethod
    def _construct_predicate(
        predicate: Optional[str],
        object_id: str,
        context: Dict[str, Any],
        is_step: bool,
        all_good: bool = True,
    ) -> Tuple[Optional[Predicate], bool]:
        rendered_template = Template(predicate).render(context)
        try:
            pred = Predicate(rendered_template)
        except (TypeError, Exception) as exception:
            if is_step:
                structlogger.error(
                    "validator.verify_predicates.step_predicate.error",
                    step=object_id,
                    exception=exception,
                    event_info=(
                        f"Could not initialize the predicate found under step "
                        f"'{object_id}': {exception}"
                    ),
                )
            else:
                structlogger.error(
                    "validator.verify_predicates.flow_guard_predicate.error",
                    flow=object_id,
                    exception=exception,
                    event_info=(
                        f"Could not initialize the predicate found in flow guard "
                        f"for flow: '{object_id}': {exception}."
                    ),
                )
            pred = None
            all_good = False

        return pred, all_good

    def _extract_predicate_syntax_tree(self, predicate: Predicate) -> Any:
        """Extract the predicate syntax tree from the given predicate.

        Args:
            predicate: The predicate from which to extract the syntax tree.

        Returns:
            The extracted syntax tree.
        """
        if isinstance(predicate.ast, NegateOperator):
            return predicate.ast.left
        return predicate.ast

    def _extract_slot_name_and_slot_value(
        self,
        predicate_syntax_tree: Any,
    ) -> tuple:
        """Extract the slot name and slot value from the predicate syntax tree.

        Args:
            predicate_syntax_tree: The predicate syntax tree.

        Returns:
            A tuple containing the slot name and slot value.
        """
        try:
            if isinstance(predicate_syntax_tree.left, Literal):
                slot_name = predicate_syntax_tree.left.value.split(".")
                slot_value = predicate_syntax_tree.right.value
            else:
                slot_name = predicate_syntax_tree.right.value.split(".")
                slot_value = predicate_syntax_tree.left.value
        except AttributeError:
            # predicate only has negation and doesn't need to be checked further
            return None, None
        return slot_name, slot_value

    def _validate_categorical_value_check(
        self,
        slot_name: str,
        slot_value: Any,
        valid_slot_values: List[str],
        all_good: bool,
        step_id: str,
        link_condition: str,
        flow_id: str,
    ) -> bool:
        """Validates the categorical slot check.

        Validates that the categorical slot is checked against valid values.

        Args:
            slot_name: name of the slot to be checked
            slot_value: value of the slot to be checked
            valid_slot_values: valid values for the given slot
            all_good: flag whether all the validations have passed so far
            step_id: id of the step in which the values are being checked
            link_condition: condition where the values are being checked
            flow_id: id of the flow where the values are being checked

        Returns:
            False, if validation failed, previous value of all_good, otherwise
        """
        valid_slot_values.append(None)
        # slot_value can either be None, a string or a list of Literal objects
        if slot_value is None:
            slot_value = [None]
        if isinstance(slot_value, str):
            slot_value = [Literal(slot_value)]

        slot_values_validity = [
            sv is None
            or re.sub(r'^[\'"](.+)[\'"]$', r"\1", sv.value) in valid_slot_values
            for sv in slot_value
        ]
        if not all(slot_values_validity):
            invalid_slot_values = [
                sv
                for (sv, slot_value_valid) in zip(slot_value, slot_values_validity)
                if not slot_value_valid
            ]
            structlogger.error(
                "validator.verify_predicates.link.invalid_condition",
                step=step_id,
                link=link_condition,
                flow=flow_id,
                event_info=(
                    f"Detected invalid condition '{link_condition}' "
                    f"at step '{step_id}' for flow id '{flow_id}'. "
                    f"Values {invalid_slot_values} are not valid values "
                    f"for slot {slot_name}. "
                    f"Please make sure that all conditions are valid."
                ),
            )
            return False
        return all_good

    def _validate_categorical_and_boolean_values_check(
        self,
        predicate: Predicate,
        all_good: bool,
        step_id: str,
        link_condition: str,
        flow_id: str,
    ) -> bool:
        """Validates the categorical and boolean slot checks.

        Validates that the categorical and boolean slots
        are checked against valid values.

        Args:
            predicate: condition that is supposed to be validated
            all_good: flag whether all the validations have passed so far
            step_id: id of the step in which the values are being checked
            link_condition: condition where the values are being checked
            flow_id: id of the flow where the values are being checked

        Returns:
            False, if validation failed, previous value of all_good, otherwise
        """
        predicate_syntax_tree = self._extract_predicate_syntax_tree(predicate)
        slot_name, slot_value = self._extract_slot_name_and_slot_value(
            predicate_syntax_tree
        )

        if slot_name is None:
            return all_good

        if slot_name[0] == "slots":
            slot_name = slot_name[1]
            # slots.{{context.variable}} gets evaluated to `slots.None`,
            # these predicates can only be validated during runtime
            if slot_name == "None":
                return all_good
        else:
            return all_good

        try:
            slot = next(slot for slot in self.domain.slots if slot.name == slot_name)
        except StopIteration:
            structlogger.error(
                "validator.verify_predicates.link.invalid_condition",
                step=step_id,
                link=link_condition,
                flow=flow_id,
                event_info=(
                    f"Detected invalid condition '{link_condition}' "
                    f"at step '{step_id}' for flow id '{flow_id}'. "
                    f"Slot {slot_name} is not defined in the domain file. "
                    f"Please make sure that all conditions are valid."
                ),
            )
            return False
        if isinstance(slot, CategoricalSlot):
            return self._validate_categorical_value_check(
                slot_name,
                slot_value,
                slot.values,
                all_good,
                step_id,
                link_condition,
                flow_id,
            )

        if (
            isinstance(slot, BooleanSlot)
            and isinstance(predicate_syntax_tree, CompareOperator)
            and not isinstance(predicate_syntax_tree.right.value, bool)
        ):
            structlogger.error(
                "validator.verify_predicates.link.invalid_condition",
                step=step_id,
                link=link_condition,
                flow=flow_id,
                event_info=(
                    f"Detected invalid condition '{link_condition}' "
                    f"at step '{step_id}' for flow id '{flow_id}'. "
                    f"Boolean slots can only be compared to "
                    f"boolean values (true, false). "
                    f"Please make sure that all conditions are valid."
                ),
            )
            return False
        return all_good

    def verify_predicates(self) -> bool:
        """Validate predicates used in flow step links and slot rejections."""
        all_good = True
        context = self._build_context()

        for flow in self.flows.underlying_flows:
            if flow.guard_condition:
                predicate, all_good = self._construct_predicate(
                    flow.guard_condition,
                    flow.id,
                    context,
                    is_step=False,
                    all_good=all_good,
                )
                if predicate and not predicate.is_valid():
                    structlogger.error(
                        "validator.verify_predicates.flow_guard.invalid_condition",
                        flow=flow.id,
                        flow_guard=flow.guard_condition,
                        event_info=(
                            f"Detected invalid flow guard condition "
                            f"'{flow.guard_condition}' for flow id '{flow.id}'. "
                            f"Please make sure that all conditions are valid."
                        ),
                    )
                    all_good = False
            for step in flow.steps_with_calls_resolved:
                for link in step.next.links:
                    if isinstance(link, IfFlowStepLink):
                        all_good = self._verify_namespaces(
                            link.condition, step.id, flow.id, all_good
                        )

                        predicate, all_good = self._construct_predicate(
                            link.condition,
                            step.id,
                            context,
                            is_step=True,
                            all_good=all_good,
                        )
                        if predicate and not predicate.is_valid():
                            structlogger.error(
                                "validator.verify_predicates.link.invalid_condition",
                                step=step.id,
                                link=link.condition,
                                flow=flow.id,
                                event_info=(
                                    f"Detected invalid condition '{link.condition}' "
                                    f"at step '{step.id}' for flow id '{flow.id}'. "
                                    f"Please make sure that all conditions are valid."
                                ),
                            )
                            all_good = False

                        all_good = self._validate_categorical_and_boolean_values_check(
                            predicate,
                            all_good=all_good,
                            step_id=step.id,
                            link_condition=link.condition,
                            flow_id=flow.id,
                        )

                if isinstance(step, CollectInformationFlowStep):
                    predicates = [predicate.if_ for predicate in step.rejections]
                    for predicate in predicates:
                        all_good = self._verify_namespaces(
                            predicate, step.id, flow.id, all_good
                        )

                        pred, all_good = self._construct_predicate(
                            predicate, step.id, context, is_step=True, all_good=all_good
                        )
                        if pred and not pred.is_valid():
                            structlogger.error(
                                "validator.verify_predicates.invalid_rejection",
                                step=step.id,
                                rejection=predicate,
                                flow=flow.id,
                                event_info=(
                                    f"Detected invalid rejection '{predicate}' "
                                    f"at `collect` step '{step.id}' "
                                    f"for flow id '{flow.id}'. "
                                    f"Please make sure that all conditions are valid."
                                ),
                            )
                            all_good = False
        return all_good

    def _verify_namespaces(
        self, predicate: str, step_id: str, flow_id: str, all_good: bool
    ) -> bool:
        slots = re.findall(r"\bslots\.\w+", predicate)
        results: List[bool] = [all_good]

        if slots:
            domain_slots = {slot.name: slot for slot in self.domain.slots}
            for slot in slots:
                slot_name = slot.split(".")[1]
                if slot_name not in domain_slots:
                    structlogger.error(
                        "validator.verify_namespaces.invalid_slot",
                        slot=slot_name,
                        step=step_id,
                        flow=flow_id,
                        event_info=(
                            f"Detected invalid slot '{slot_name}' "
                            f"at step '{step_id}' "
                            f"for flow id '{flow_id}'. "
                            f"Please make sure that all slots are specified "
                            f"in the domain file."
                        ),
                    )
                    results.append(False)

        if not slots:
            # no slots found, check if context namespace is used
            variables = re.findall(r"\bcontext\.\w+", predicate)
            if not variables:
                structlogger.error(
                    "validator.verify_namespaces"
                    ".referencing_variables_without_namespace",
                    step=step_id,
                    predicate=predicate,
                    flow=flow_id,
                    event_info=(
                        f"Predicate '{predicate}' at step '{step_id}' for flow id "
                        f"'{flow_id}' references one or more variables  without "
                        f"the `slots.` or `context.` namespace prefix. "
                        f"Please make sure that all variables reference the required "
                        f"namespace."
                    ),
                )
                results.append(False)

        return all(results)

    def verify_flows(self) -> bool:
        """Checks for inconsistencies across flows."""
        structlogger.info("validation.flows.started")

        if self.flows.is_empty():
            structlogger.warn(
                "validator.verify_flows",
                event_info=(
                    "No flows were found in the data files. "
                    "Will not proceed with flow validation."
                ),
            )
            return True

        # add all flow validation conditions here
        flow_validation_conditions = [
            self.verify_flows_steps_against_domain(),
            self.verify_unique_flows(),
            self.verify_predicates(),
        ]

        all_good = all(flow_validation_conditions)

        structlogger.info("validation.flows.ended")

        return all_good

    def validate_button_payloads(self) -> bool:
        """Check if the response button payloads are valid."""
        all_good = True
        for utter_name, response in self.domain.responses.items():
            for variation in response:
                for button in variation.get("buttons", []):
                    payload = button.get("payload")
                    if payload is None:
                        structlogger.error(
                            "validator.validate_button_payloads.missing_payload",
                            event_info=(
                                f"The button '{button.get('title')}' in response "
                                f"'{utter_name}' does not have a payload. "
                                f"Please add a payload to the button."
                            ),
                        )
                        all_good = False
                        continue

                    if not payload.strip():
                        structlogger.error(
                            "validator.validate_button_payloads.empty_payload",
                            event_info=(
                                f"The button '{button.get('title')}' in response "
                                f"'{utter_name}' has an empty payload. "
                                f"Please add a payload to the button."
                            ),
                        )
                        all_good = False
                        continue

                    regex_reader = create_regex_pattern_reader(
                        UserMessage(text=payload), self.domain
                    )

                    if regex_reader is None:
                        structlogger.warning(
                            "validator.validate_button_payloads.free_form_string",
                            event_info=(
                                "Using a free form string in payload of a button "
                                "implies that the string will be sent to the NLU "
                                "interpreter for parsing. To avoid the need for "
                                "parsing at runtime, it is recommended to use one "
                                "of the documented formats "
                                "(https://rasa.com/docs/rasa-pro/concepts/responses#buttons)"
                            ),
                        )
                        continue

                    if isinstance(
                        regex_reader, CommandPayloadReader
                    ) and regex_reader.is_above_slot_limit(payload):
                        structlogger.error(
                            "validator.validate_button_payloads.slot_limit_exceeded",
                            event_info=(
                                f"The button '{button.get('title')}' in response "
                                f"'{utter_name}' has a payload that sets more than "
                                f"{MAX_NUMBER_OF_SLOTS} slots. "
                                f"Please make sure that the number of slots set by "
                                f"the button payload does not exceed the limit."
                            ),
                        )
                        all_good = False
                        continue

                    if isinstance(regex_reader, YAMLStoryReader):
                        # the payload could contain double curly braces
                        # we need to remove 1 set of curly braces
                        payload = payload.replace("{{", "{").replace("}}", "}")

                    resulting_message = regex_reader.unpack_regex_message(
                        message=Message(data={"text": payload}), domain=self.domain
                    )

                    if not (
                        resulting_message.has_intent()
                        or resulting_message.has_commands()
                    ):
                        structlogger.error(
                            "validator.validate_button_payloads.invalid_payload_format",
                            event_info=(
                                f"The button '{button.get('title')}' in response "
                                f"'{utter_name}' does not follow valid payload formats "
                                f"for triggering a specific intent and entities or for "
                                f"triggering a SetSlot command."
                            ),
                            calm_docs_link=DOCS_URL_RESPONSES + "#payload-syntax",
                            nlu_docs_link=DOCS_URL_RESPONSES
                            + "#triggering-intents-or-passing-entities",
                        )
                        all_good = False

                    if resulting_message.has_commands():
                        # validate that slot names are unique
                        slot_names = set()
                        for command in resulting_message.get(COMMANDS, []):
                            slot_name = command.get("name")
                            if slot_name and slot_name in slot_names:
                                structlogger.error(
                                    "validator.validate_button_payloads.duplicate_slot_name",
                                    event_info=(
                                        f"The button '{button.get('title')}' "
                                        f"in response '{utter_name}' has a "
                                        f"command to set the slot "
                                        f"'{slot_name}' multiple times. "
                                        f"Please make sure that each slot "
                                        f"is set only once."
                                    ),
                                )
                                all_good = False
                            slot_names.add(slot_name)

        return all_good

    def validate_CALM_slot_mappings(self) -> bool:
        """Check if the usage of slot mappings in a CALM assistant is valid."""
        all_good = True

        for slot in self.domain._user_slots:
            nlu_mappings = any(
                [
                    SlotMappingType(
                        mapping.get("type", SlotMappingType.FROM_LLM.value)
                    ).is_predefined_type()
                    for mapping in slot.mappings
                ]
            )
            llm_mappings = any(
                [
                    SlotMappingType(mapping.get("type", SlotMappingType.FROM_LLM.value))
                    == SlotMappingType.FROM_LLM
                    for mapping in slot.mappings
                ]
            )
            custom_mappings = any(
                [
                    SlotMappingType(mapping.get("type", SlotMappingType.FROM_LLM.value))
                    == SlotMappingType.CUSTOM
                    for mapping in slot.mappings
                ]
            )

            all_good = self._slot_contains_all_mappings_types(
                llm_mappings, nlu_mappings, custom_mappings, slot.name, all_good
            )

            all_good = self._custom_action_name_is_defined_in_the_domain(
                custom_mappings, slot, all_good
            )

            all_good = self._config_contains_nlu_command_adapter(
                nlu_mappings, slot.name, all_good
            )

            all_good = self._uses_from_llm_mappings_in_a_NLU_based_assistant(
                llm_mappings, slot.name, all_good
            )

        return all_good

    @staticmethod
    def _slot_contains_all_mappings_types(
        llm_mappings: bool,
        nlu_mappings: bool,
        custom_mappings: bool,
        slot_name: str,
        all_good: bool,
    ) -> bool:
        if llm_mappings and (nlu_mappings or custom_mappings):
            structlogger.error(
                "validator.validate_slot_mappings_in_CALM.llm_and_nlu_mappings",
                slot_name=slot_name,
                event_info=(
                    f"The slot '{slot_name}' has both LLM and "
                    f"NLU or custom slot mappings. "
                    f"Please make sure that the slot has only one type of mapping."
                ),
                docs_link=DOCS_URL_DOMAIN + "#calm-slot-mappings",
            )
            all_good = False

        return all_good

    def _custom_action_name_is_defined_in_the_domain(
        self,
        custom_mappings: bool,
        slot: Slot,
        all_good: bool,
    ) -> bool:
        if not custom_mappings:
            return all_good

        if not self.flows:
            return all_good

        is_custom_action_defined = any(
            [
                mapping.get("action") is not None
                and mapping.get("action") in self.domain.action_names_or_texts
                for mapping in slot.mappings
            ]
        )

        if is_custom_action_defined:
            return all_good

        slot_collected_by_flows = any(
            [
                step.collect == slot.name
                for flow in self.flows.underlying_flows
                for step in flow.steps
                if isinstance(step, CollectInformationFlowStep)
            ]
        )

        if not slot_collected_by_flows:
            # if the slot is not collected by any flow,
            # it could be a DM1 custom slot
            return all_good

        custom_action_ask_name = f"action_ask_{slot.name}"
        if custom_action_ask_name not in self.domain.action_names_or_texts:
            structlogger.error(
                "validator.validate_slot_mappings_in_CALM.custom_action_not_in_domain",
                slot_name=slot.name,
                event_info=(
                    f"The slot '{slot.name}' has a custom slot mapping, but "
                    f"neither the action '{custom_action_ask_name}' nor "
                    f"another custom action are defined in the domain file. "
                    f"Please add one of the actions to your domain file."
                ),
                docs_link=DOCS_URL_DOMAIN + "#custom-slot-mappings",
            )
            all_good = False

        return all_good

    def _config_contains_nlu_command_adapter(
        self, nlu_mappings: bool, slot_name: str, all_good: bool
    ) -> bool:
        if not nlu_mappings:
            return all_good

        if not self.flows:
            return all_good

        contains_nlu_command_adapter = any(
            [
                component.get("name") == "NLUCommandAdapter"
                for component in self.config.get("pipeline", [])
            ]
        )

        if not contains_nlu_command_adapter:
            structlogger.error(
                "validator.validate_slot_mappings_in_CALM.nlu_mappings_without_adapter",
                slot_name=slot_name,
                event_info=(
                    f"The slot '{slot_name}' has NLU slot mappings, "
                    f"but the NLUCommandAdapter is not present in the "
                    f"pipeline. Please add the NLUCommandAdapter to the "
                    f"pipeline in the config file."
                ),
                docs_link=DOCS_URL_DOMAIN + "#nlu-based-predefined-slot-mappings",
            )
            all_good = False

        return all_good

    def _uses_from_llm_mappings_in_a_NLU_based_assistant(
        self, llm_mappings: bool, slot_name: str, all_good: bool
    ) -> bool:
        if not llm_mappings:
            return all_good

        if self.flows:
            return all_good

        structlogger.error(
            "validator.validate_slot_mappings_in_CALM.llm_mappings_without_flows",
            slot_name=slot_name,
            event_info=(
                f"The slot '{slot_name}' has LLM slot mappings, "
                f"but no flows are present in the training data files. "
                f"Please add flows to the training data files."
            ),
        )
        return False
