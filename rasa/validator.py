import logging
import re
import string
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Text, Tuple

import jinja2.exceptions
import structlog
from jinja2 import Template
from pypred.ast import CompareOperator, Literal, NegateOperator

import rasa.core.training.story_conflict
import rasa.shared.nlu.constants
from rasa.agents.validation import validate_agent_names_not_conflicting_with_flows
from rasa.core.channels import UserMessage
from rasa.core.config.configuration import Configuration
from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.engine.language import Language
from rasa.exceptions import ValidationError
from rasa.shared.constants import (
    ASSISTANT_ID_DEFAULT_VALUE,
    ASSISTANT_ID_KEY,
    CONFIG_ADDITIONAL_LANGUAGES_KEY,
    CONFIG_LANGUAGE_KEY,
    CONFIG_MANDATORY_KEYS,
    CONFIG_PIPELINE_KEY,
    DOCS_URL_ACTIONS,
    DOCS_URL_DOMAIN,
    DOCS_URL_DOMAINS,
    DOCS_URL_FORMS,
    DOCS_URL_RESPONSES,
    REQUIRED_SLOTS_KEY,
    RESPONSE_CONDITION,
    UTTER_PREFIX,
)
from rasa.shared.core import constants
from rasa.shared.core.command_payload_reader import (
    MAX_NUMBER_OF_SLOTS,
    CommandPayloadReader,
)
from rasa.shared.core.constants import (
    KEY_ALLOW_NLU_CORRECTION,
    SLOTS,
    SlotMappingType,
)
from rasa.shared.core.domain import (
    RESPONSE_KEYS_TO_INTERPOLATE,
    Domain,
)
from rasa.shared.core.events import ActionExecuted, ActiveLoop, UserUttered
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.constants import KEY_NAME, KEY_TRANSLATION
from rasa.shared.core.flows.flow_step_links import IfFlowStepLink
from rasa.shared.core.flows.steps.action import ActionFlowStep
from rasa.shared.core.flows.steps.collect import CollectInformationFlowStep
from rasa.shared.core.flows.steps.link import LinkFlowStep
from rasa.shared.core.flows.steps.set_slots import SetSlotsFlowStep
from rasa.shared.core.flows.utils import (
    get_duplicate_slot_persistence_config_error_message,
    get_invalid_slot_persistence_config_error_message,
    warn_deprecated_collect_step_config,
)
from rasa.shared.core.generator import TrainingDataGenerator
from rasa.shared.core.slot_mappings import CoexistenceSystemType
from rasa.shared.core.slots import BooleanSlot, CategoricalSlot, ListSlot, Slot
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.data import create_regex_pattern_reader
from rasa.shared.exceptions import RasaException
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.constants import COMMANDS, INTENT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.telemetry import track_validation_error_log
from rasa.utils.pypred import Predicate

logger = logging.getLogger(__name__)

structlog_processors = structlog.get_config()["processors"]
updated_processors = [track_validation_error_log] + structlog_processors
structlogger = structlog.get_logger(processors=updated_processors)


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

        nlu_data_intents = {e.data[INTENT] for e in self.intents.intent_examples}

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
            duplication_hash[text].add(example.get(INTENT))

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
                        f"The intent '{intent}' is not used in any story, rule or flow."
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

    def check_for_no_empty_parenthesis_in_responses(self) -> bool:
        """Checks if there are no empty parenthesis in utterances."""
        everything_is_alright = True

        for response_text, response_variations in self.domain.responses.items():
            if not response_variations:
                structlogger.error(
                    "validator.empty_response",
                    response=response_text,
                    event_info=(
                        f"The response '{response_text}' in the domain file "
                        f"does not have any variations. Please add at least one "
                        f"variation to the response."
                    ),
                )
                everything_is_alright = False

            for response in response_variations:
                if any(
                    self.check_for_placeholder(response.get(key))
                    for key in RESPONSE_KEYS_TO_INTERPOLATE
                ):
                    structlogger.error(
                        "validator.empty_parenthesis_in_utterances",
                        response=response_text,
                        event_info=(
                            f"The response '{response_text}' in the domain file "
                            f"contains empty parenthesis, which is not permitted. "
                            f"Please remove the empty parenthesis."
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
                for condition in mapping.conditions:
                    condition_active_loop = condition.active_loop
                    mapping_type = mapping.type
                    if (
                        condition_active_loop
                        and condition_active_loop not in self.domain.form_names
                    ):
                        structlogger.error(
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
                        f"The config file is missing the '{key}' mandatory key."
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
        flow_id: str,
    ) -> bool:
        """Validates that a collect step can have either an action or an utterance.

        Also logs an error if neither an action nor an utterance is defined.

        Args:
            collect: the name of the slot to collect
            all_good: boolean value indicating the validation status
            domain_slots: dictionary of domain slots
            flow_id: the ID of the flow being validated

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
                flow=flow_id,
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
                flow=flow_id,
                event_info=(
                    f"The collect step '{collect.collect}' has neither a response "
                    f"nor an action defined, nor an initial value defined in the "
                    f"domain. You can fix this by adding a response named "
                    f"'{collect.utter}' used in the collect step."
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
                            step, all_good, domain_slots, flow.id
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
            elif object_id.startswith("utter_"):
                structlogger.error(
                    "validator.validate_conditional_response_variation_predicates.error",
                    utter=object_id,
                    exception=exception,
                    event_info=(
                        f"Could not initialize the predicate found under response "
                        f"variation '{object_id}': {exception}"
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
    ) -> Tuple[Optional[List[str]], Optional[Any]]:
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
            structlogger.error(
                "validator.verify_predicates.link.invalid_condition",
                step=step_id,
                link=link_condition,
                flow=flow_id,
                event_info=(
                    f"Detected invalid condition '{link_condition}' "
                    f"at step '{step_id}' for flow id '{flow_id}'. "
                    f"The condition contains invalid values for slot {slot_name}. "
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
        slot_namespace, slot_value = self._extract_slot_name_and_slot_value(
            predicate_syntax_tree
        )

        if slot_namespace is None:
            return all_good

        if slot_namespace[0] == "slots":
            slot_name = slot_namespace[1]
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
                    dumped_predicates = [predicate.if_ for predicate in step.rejections]
                    for dumped_predicate in dumped_predicates:
                        all_good = self._verify_namespaces(
                            dumped_predicate, step.id, flow.id, all_good
                        )

                        predicate, all_good = self._construct_predicate(
                            dumped_predicate,
                            step.id,
                            context,
                            is_step=True,
                            all_good=all_good,
                        )
                        if predicate and not predicate.is_valid():
                            structlogger.error(
                                "validator.verify_predicates.invalid_rejection",
                                step=step.id,
                                rejection=predicate,
                                flow=flow.id,
                                event_info=(
                                    f"Detected invalid rejection '{dumped_predicate}' "
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
            self.verify_slot_persistence_configuration(),
        ]

        all_good = all(flow_validation_conditions)

        structlogger.info("validation.flows.ended")

        return all_good

    def validate_agent_flow_conflicts(self, sub_agents_path: str) -> bool:
        """Validates that agent names don't conflict with flow names.

        Args:
            sub_agents_path: Path to the sub-agents directory.

        Returns:
            True if validation passes, False otherwise.
        """
        if self.flows.is_empty():
            return True

        try:
            configuration = Configuration.get_instance()
            available_agents = configuration.available_agents
            flow_names = {flow.id for flow in self.flows.underlying_flows}

            structlogger.debug(
                "validator.agent_flow_conflicts.debug",
                event_info="Checking conflicts between agents and flows",
            )

            validate_agent_names_not_conflicting_with_flows(
                available_agents.agents, flow_names
            )
            return True
        except Exception as e:
            structlogger.error(
                "validator.agent_flow_conflicts",
                sub_agents_path=sub_agents_path,
                error=str(e),
                event_info=f"Agent-flow name conflict validation failed: {e}",
            )
            return False

    def _get_response_translation_warnings(self) -> list:
        """Collect warnings for responses missing translations.

        Returns:
            List of warnings for responses missing translations.
        """
        additional_languages = self.config.get(CONFIG_ADDITIONAL_LANGUAGES_KEY) or []
        response_warnings = []

        for response_name, responses in self.domain.responses.items():
            provided_languages = set()
            # For each response variation, we check if the additional
            # languages are available in at least on variation
            for response in responses:
                translation = response.get(KEY_TRANSLATION) or {}
                for language_code in additional_languages:
                    if translation.get(language_code):
                        provided_languages.add(language_code)

            missing_languages = [
                lang for lang in additional_languages if lang not in provided_languages
            ]
            if missing_languages:
                language_code_str = ", ".join(missing_languages)
                response_warnings.append(
                    {
                        "event": (
                            "validator.verify_translations.missing_response_translation"
                        ),
                        "response": response_name,
                        "missing_languages": missing_languages,
                        "event_info": (
                            f"The response '{response_name}' is "
                            f"missing a translation for the following "
                            f"languages: {language_code_str}."
                        ),
                    }
                )
        return response_warnings

    def _get_flow_translation_warnings(self) -> list:
        """Collect warnings for flows missing translations.

        Returns:
            List of warnings for flows missing translations.
        """
        additional_languages = self.config.get(CONFIG_ADDITIONAL_LANGUAGES_KEY) or []

        flow_warnings = []
        for flow in self.flows.underlying_flows:
            required_field_translation = [KEY_NAME]
            missing_languages = []
            for language_code in additional_languages:
                translation = flow.translation.get(language_code)
                # If translation for the language code doesn't exist,
                # or the required fields are not set properly,
                # we add the language code to the list.
                if not translation or not all(
                    getattr(translation, field, None)
                    for field in required_field_translation
                ):
                    missing_languages.append(language_code)

            if missing_languages:
                language_code_str = ", ".join(missing_languages)
                flow_warnings.append(
                    {
                        "event": (
                            "validator.verify_translations.missing_flow_translation"
                        ),
                        "flow": flow.id,
                        "missing_languages": missing_languages,
                        "event_info": (
                            f"The flow '{flow.id}' is missing the translation for "
                            f"the following languages: {language_code_str}."
                        ),
                    }
                )
        return flow_warnings

    def verify_config_language(self) -> bool:
        """Verify that config languages are properly set up.

        Returns:
            `True` if all languages are properly set up, `False` otherwise.

        Raises:
            RasaException: If the default language is listed as an
            additional language or if the language code is invalid.
        """
        language = self.config.get(CONFIG_LANGUAGE_KEY)
        additional_languages = self.config.get(CONFIG_ADDITIONAL_LANGUAGES_KEY, [])

        # Check if the default language is in the additional languages.
        if language in additional_languages:
            raise RasaException(
                f"The default language '{language}' is listed as an additional "
                f"language in the configuration file. Please remove it from "
                f"the list of additional languages."
            )

        # Verify the language codes by initializing the Language class.
        for language_code in [language] + additional_languages:
            Language.from_language_code(language_code=language_code)

        return True

    def verify_translations(self, summary_mode: bool = False) -> bool:
        """Checks for inconsistencies in translations.

        Args:
            summary_mode: If True, logs a single aggregated warning per category;
                otherwise, logs each warning individually.

        Returns:
            `True` if no inconsistencies were found, `False` otherwise.

        Raises:
            Warning: Single warning per response or flow missing translations
            if `summary_mode` is `True`, otherwise one warning per missing translation.
        """
        all_good = self.verify_config_language()

        additional_languages = self.config.get(CONFIG_ADDITIONAL_LANGUAGES_KEY, [])
        if not additional_languages:
            return all_good

        response_warnings = self._get_response_translation_warnings()
        flow_warnings = self._get_flow_translation_warnings()

        if summary_mode:
            if response_warnings:
                count = len(response_warnings)
                structlogger.warn(
                    "validator.verify_translations.missing_response_translation_summary",
                    count=count,
                    event_info=(
                        f"{count} response{' is' if count == 1 else 's are'} "
                        f"missing translations for some languages. "
                        "Run 'rasa data validate translations' for details."
                    ),
                )
            if flow_warnings:
                count = len(flow_warnings)
                structlogger.warn(
                    "validator.verify_translations.missing_flow_translation_summary",
                    count=count,
                    event_info=(
                        f"{count} flow{' is' if count == 1 else 's are'} "
                        f"missing translations for some languages. "
                        "Run 'rasa data validate translations' for details."
                    ),
                )
        else:
            for warning in response_warnings + flow_warnings:
                structlogger.warn(**warning)

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
                [mapping.type.is_predefined_type() for mapping in slot.mappings]
            )
            llm_mappings = any(
                [mapping.type == SlotMappingType.FROM_LLM for mapping in slot.mappings]
            )
            controlled_mappings = any(
                [
                    mapping.type == SlotMappingType.CONTROLLED
                    for mapping in slot.mappings
                ]
            )

            all_good = self._allow_nlu_correction_is_valid(slot, nlu_mappings, all_good)

            all_good = self._custom_action_name_is_defined_in_the_domain(
                controlled_mappings, slot, all_good
            )

            all_good = self._validate_controlled_mappings(slot, all_good)

            all_good = self._config_contains_nlu_command_adapter(
                nlu_mappings, slot.name, all_good
            )

            all_good = self._uses_from_llm_mappings_in_a_NLU_based_assistant(
                llm_mappings, slot.name, all_good
            )

        return all_good

    @staticmethod
    def _allow_nlu_correction_is_valid(
        slot: Slot, nlu_mappings: bool, all_good: bool
    ) -> bool:
        """Verify that `allow_nlu_correction` property is used correctly in a `from_llm` mappings only."""  # noqa: E501
        if not slot.mappings:
            return all_good

        invalid_usage = False

        for mapping in slot.mappings:
            allow_nlu_correction = mapping.allow_nlu_correction
            if allow_nlu_correction and mapping.type != SlotMappingType.FROM_LLM:
                invalid_usage = True

            if allow_nlu_correction and not nlu_mappings:
                structlogger.error(
                    "validator.validate_slot_mappings_in_CALM.nlu_mappings_not_present",
                    slot_name=slot.name,
                    event_info=(
                        f"The slot '{slot.name}' does not have any "
                        f"NLU-based slot mappings. "
                        f"The property `allow_nlu_correction` is only "
                        f"applicable when the slot "
                        f"contains both NLU-based and LLM-based slot mappings."
                    ),
                )
                all_good = False

        if invalid_usage:
            structlogger.error(
                "validator.validate_slot_mappings_in_CALM.allow_nlu_correction",
                slot_name=slot.name,
                event_info=(
                    f"The slot '{slot.name}' has at least 1 slot mapping with "
                    f"'{KEY_ALLOW_NLU_CORRECTION}' set to 'true', but "
                    f"the slot mapping type is not 'from_llm'. "
                    f"Please set the slot mapping type to 'from_llm' "
                    f"to allow the LLM to correct this slot."
                ),
            )
            all_good = False

        return all_good

    def _custom_action_name_is_defined_in_the_domain(
        self,
        controlled_mappings: bool,
        slot: Slot,
        all_good: bool,
    ) -> bool:
        if not controlled_mappings:
            return all_good

        for mapping in slot.mappings:
            if (
                mapping.run_action_every_turn is not None
                and mapping.run_action_every_turn
                not in self.domain.action_names_or_texts
            ):
                structlogger.error(
                    "validator.validate_slot_mappings_in_CALM.custom_action_not_in_domain",
                    slot_name=slot.name,
                    action_name=mapping.run_action_every_turn,
                    event_info=(
                        f"The slot '{slot.name}' has a custom action "
                        f"'{mapping.run_action_every_turn}' "
                        f"defined in its slot mappings, "
                        f"but the action is not listed in the domain actions. "
                        f"Please add the action to your domain file."
                    ),
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
                for component in self.config.get(CONFIG_PIPELINE_KEY, [])
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

    def verify_slot_persistence_configuration(self) -> bool:
        """Verifies the validity of slot persistence after flow ends configuration.

        Only slots used in either a collect step or a set_slot step can be persisted and
        the configuration can either set at the flow level or the collect step level,
        but not both.

        Returns:
            bool: True if all slot persistence configuration is valid, False otherwise.

        Raises:
            DeprecationWarning: If reset_after_flow_ends is used in collect steps.
        """
        all_good = True

        for flow in self.flows.underlying_flows:
            flow_id = flow.id
            persist_slots = flow.persisted_slots
            has_flow_level_persistence = True if persist_slots else False
            flow_slots = set()

            for step in flow.steps_with_calls_resolved:
                if isinstance(step, SetSlotsFlowStep):
                    flow_slots.update([slot["key"] for slot in step.slots])

                elif isinstance(step, CollectInformationFlowStep):
                    collect_step = step.collect
                    flow_slots.add(collect_step)
                    if not step.reset_after_flow_ends:
                        warn_deprecated_collect_step_config()

                        if has_flow_level_persistence:
                            structlogger.error(
                                "validator.verify_slot_persistence_configuration.duplicate_config",
                                flow=flow_id,
                                collect_step=collect_step,
                                event_info=get_duplicate_slot_persistence_config_error_message(
                                    flow_id, collect_step
                                ),
                            )
                            all_good = False

            if has_flow_level_persistence:
                if not self._is_persist_slots_valid(persist_slots, flow_slots, flow_id):
                    all_good = False
        return all_good

    def _is_persist_slots_valid(
        self, persist_slots: List[str], flow_slots: Set[str], flow_id: str
    ) -> bool:
        invalid_slots = set(persist_slots) - flow_slots
        is_valid = False if invalid_slots else True

        if invalid_slots:
            structlogger.error(
                "validator.verify_slot_persistence_configuration.invalid_persist_slot",
                flow=flow_id,
                event_info=get_invalid_slot_persistence_config_error_message(
                    flow_id, invalid_slots
                ),
            )
        return is_valid

    def verify_studio_supported_validations(self) -> bool:
        """Validates the assistant project for Rasa Studio supported features.

        Ensure to add new validations here if they are required for
        Rasa Studio Upload CLI.
        """
        if self.domain.is_empty():
            structlogger.error(
                "rasa.validator.verify_studio_supported_validations.empty_domain",
                event_info="Encountered empty domain during validation.",
            )
            raise ValidationError(
                code="rasa.validator.verify_studio_supported_validations.empty_domain",
                event_info="Encountered empty domain during validation.",
            )

        self.warn_if_config_mandatory_keys_are_not_set()

        valid_responses = (
            self.check_for_no_empty_parenthesis_in_responses()
            and self.validate_button_payloads()
        )
        valid_nlu = self.verify_nlu()
        valid_flows = all(
            [
                self.verify_flows_steps_against_domain(),
                self.verify_unique_flows(),
                self.verify_predicates(),
            ]
        )
        valid_translations = self.verify_translations(summary_mode=True)
        valid_calm_slot_mappings = self.validate_CALM_slot_mappings()

        all_good = (
            valid_responses
            and valid_nlu
            and valid_flows
            and valid_translations
            and valid_calm_slot_mappings
        )

        return all_good

    def verify_slot_validation(self) -> bool:
        """Validates the slot validation configuration in the domain file."""
        all_good = True

        for slot in self.domain._user_slots:
            if slot.requires_validation():
                refill_utter = slot.validation.refill_utter  # type: ignore[union-attr]
                if refill_utter and refill_utter not in self.domain.responses:
                    self._log_slot_validation_error(
                        slot.name, "refill utterance", refill_utter
                    )
                    all_good = False
                rejections = slot.validation.rejections  # type: ignore[union-attr]
                for rejection in rejections:
                    if rejection.utter not in self.domain.responses:
                        self._log_slot_validation_error(
                            slot.name, "rejection utterance", rejection.utter
                        )
                        all_good = False

        return all_good

    def _log_slot_validation_error(self, slot_name: str, key: str, value: str) -> None:
        structlogger.error(
            "validator.verify_slot_validation.response_not_in_domain",
            slot=slot_name,
            event_info=(
                f"The slot '{slot_name}' requires validation, "
                f"but the {key} '{value}' "
                f"is not listed in the domain responses. "
                f"Please add it to your domain file."
            ),
        )

    @staticmethod
    def _validate_controlled_mappings(slot: Slot, all_good: bool) -> bool:
        for mapping in slot.mappings:
            if (
                mapping.run_action_every_turn is not None
                and mapping.type != SlotMappingType.CONTROLLED
            ):
                structlogger.error(
                    "validator.validate_slot_mappings_in_CALM.run_action_every_turn_invalid",
                    slot_name=slot.name,
                    event_info=(
                        f"The slot '{slot.name}' has a custom action "
                        f"'{mapping.run_action_every_turn}' "
                        f"defined in its slot mapping, "
                        f"but the slot mapping type is not 'controlled'. "
                    ),
                )
                all_good = False

            if (
                mapping.coexistence_system is not None
                and mapping.type != SlotMappingType.CONTROLLED
            ):
                structlogger.error(
                    "validator.validate_slot_mappings_in_CALM.coexistence_system_invalid",
                    slot_name=slot.name,
                    event_info=(
                        f"The slot '{slot.name}' has a coexistence system "
                        f"'{mapping.coexistence_system.value}' "
                        f"defined in its slot mapping, "
                        f"but the slot mapping type is not 'controlled'. "
                    ),
                )
                all_good = False

            if (
                mapping.coexistence_system is not None
                and mapping.coexistence_system != CoexistenceSystemType.SHARED
                and slot.shared_for_coexistence
            ):
                structlogger.error(
                    "validator.validate_slot_mappings_in_CALM.shared_for_coexistence_invalid",
                    slot_name=slot.name,
                    event_info=(
                        f"The slot '{slot.name}' has the `shared_for_coexistence` "
                        f"property set to `True`, but the slot mapping `controlled` "
                        f"type defines the `coexistence_system` property with a "
                        f"value different to the expected `SHARED` value. "
                    ),
                )
                all_good = False

        multiple_controlled_mappings = {
            mapping.coexistence_system.value
            for mapping in slot.mappings
            if mapping.type == SlotMappingType.CONTROLLED
            and mapping.coexistence_system is not None
        }
        contains_inconsistent_coexistence_system = len(multiple_controlled_mappings) > 1

        if contains_inconsistent_coexistence_system:
            structlogger.error(
                "validator.validate_slot_mappings_in_CALM.inconsistent_multiple_mappings",
                slot_name=slot.name,
                event_info=(
                    f"The slot '{slot.name}' has multiple `controlled` mappings "
                    f"with different coexistence systems defined: "
                    f"'{sorted(list(multiple_controlled_mappings))}'. "
                    f"Please only define one coexistence system for the slot. "
                ),
            )
            all_good = False

        return all_good

    def verify_prompt_templates(self) -> bool:
        """Verify that all prompt templates have valid Jinja2 syntax.

        Returns:
            True if all templates are valid, False otherwise.
        """
        all_good = True

        # Check the components in the pipeline and policies for prompt templates
        pipeline = self.config.get(CONFIG_PIPELINE_KEY, [])
        for component in pipeline:
            if isinstance(component, dict):
                component_name = component.get("name", "")
                prompt_template = component.get("prompt_template")
                if prompt_template:
                    all_good = (
                        self._validate_template_file(
                            prompt_template, component_name, "pipeline component"
                        )
                        and all_good
                    )

        # Check policies for prompt templates
        policies = self.config.get("policies") or []
        for policy in policies:
            if isinstance(policy, dict):
                policy_name = policy.get("name", "")
                prompt_template = policy.get("prompt_template")
                if prompt_template:
                    all_good = (
                        self._validate_template_file(
                            prompt_template, policy_name, "policy"
                        )
                        and all_good
                    )

        return all_good

    def _validate_template_file(
        self, prompt_template: str, component_name: str, component_type: str
    ) -> bool:
        """Validate a single prompt template file.

        Args:
            prompt_template: The template file path to validate
            component_name: Name of the component using the template
            component_type: Type of component (e.g., "policy", "pipeline component")

        Returns:
            True if template is valid, False otherwise.
        """
        try:
            # Use a simple default template, as we're assuming
            # that the default templates are valid
            default_template = "{{ content }}"
            template_content = rasa.shared.utils.llm.get_prompt_template(
                prompt_template,
                default_template,
                log_source_component=f"validator.{component_name}",
                log_source_method="init",
            )

            # Validate Jinja2 syntax using the shared validation function
            rasa.shared.utils.llm.validate_jinja2_template(template_content)
            return True
        except jinja2.exceptions.TemplateSyntaxError as e:
            structlogger.error(
                "validator.verify_prompt_templates.syntax_error",
                component=component_name,
                component_type=component_type,
                event_info=(
                    f"Invalid Jinja2 template syntax in file {prompt_template} "
                    f"at line {e.lineno}: {e.message}"
                ),
                error=str(e),
                template_line=e.lineno,
                template_file=prompt_template,
            )
            return False
        except Exception as e:
            structlogger.error(
                "validator.verify_prompt_templates.error",
                component=component_name,
                component_type=component_type,
                event_info=f"Error validating prompt template: {e}",
                error=str(e),
            )
            return False

    def validate_conditional_response_variation_predicates(self) -> bool:
        """Validate the conditional response variation predicates."""
        context = {SLOTS: {slot.name: None for slot in self.domain.slots}}
        all_good = True

        for utter_name, variations in self.domain.responses.items():
            for variation in variations:
                condition = variation.get(RESPONSE_CONDITION)
                if not isinstance(condition, str):
                    continue

                predicate, all_good = self._construct_predicate(
                    condition,
                    utter_name,
                    context,
                    is_step=False,
                    all_good=all_good,
                )
                if not predicate:
                    continue

                if not predicate.is_valid():
                    structlogger.error(
                        "validator.validate_conditional_response_variation_predicates.invalid_condition",
                        utter=utter_name,
                        event_info=(
                            f"Detected invalid condition '{condition}' "
                            f"for response variation '{utter_name}'. "
                            f"Please make sure that all conditions are valid."
                        ),
                    )
                    all_good = False
                    continue

                predicate_syntax_tree = self._extract_predicate_syntax_tree(predicate)
                slot_namespace, _ = self._extract_slot_name_and_slot_value(
                    predicate_syntax_tree
                )

                if slot_namespace is not None and slot_namespace[0] != SLOTS:
                    structlogger.error(
                        "validator.validate_conditional_response_variation_predicates.invalid_namespace",
                        utter=utter_name,
                        event_info=(
                            f"Detected invalid namespace '{slot_namespace[0]}' in "
                            f"condition '{condition}' for response variation "
                            f"'{utter_name}'. Please make sure that you're "
                            f"using a valid namespace. "
                            f"The current supported option is: 'slots'."
                        ),
                    )
                    all_good = False
                    continue

                if (
                    slot_namespace is not None
                    and slot_namespace[1] not in context[SLOTS]
                ):
                    structlogger.error(
                        "validator.validate_conditional_response_variation_predicates.invalid_slot",
                        utter=utter_name,
                        event_info=(
                            f"Detected invalid slot '{slot_namespace[0]}' in "
                            f"condition '{condition}' for response variation "
                            f"'{utter_name}'. Please make sure that all slots "
                            f"are specified in the domain file."
                        ),
                    )
                    all_good = False

        return all_good
