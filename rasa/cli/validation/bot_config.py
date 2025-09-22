import argparse
from typing import TYPE_CHECKING, Optional

import structlog

from rasa import telemetry
from rasa.exceptions import ValidationError
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.common import display_research_study_prompt

if TYPE_CHECKING:
    from rasa.validator import Validator

structlogger = structlog.get_logger()

FREE_TEXT_INPUT_PROMPT = "Type out your own message..."


def _validate_domain(validator: "Validator") -> bool:
    valid_domain_validity = validator.verify_domain_validity()
    valid_actions_in_stories_rules = validator.verify_actions_in_stories_rules()
    valid_forms_in_stories_rules = validator.verify_forms_in_stories_rules()
    valid_form_slots = validator.verify_form_slots()
    valid_slot_mappings = validator.verify_slot_mappings()
    valid_responses = validator.check_for_no_empty_parenthesis_in_responses()
    valid_buttons = validator.validate_button_payloads()
    valid_slot_validation = validator.verify_slot_validation()
    valid_conditional_responses = (
        validator.validate_conditional_response_variation_predicates()
    )
    return (
        valid_domain_validity
        and valid_actions_in_stories_rules
        and valid_forms_in_stories_rules
        and valid_form_slots
        and valid_slot_mappings
        and valid_responses
        and valid_buttons
        and valid_slot_validation
        and valid_conditional_responses
    )


def _validate_nlu(validator: "Validator", fail_on_warnings: bool) -> bool:
    return validator.verify_nlu(not fail_on_warnings)


def _validate_story_structure(
    validator: "Validator", max_history: Optional[int], fail_on_warnings: bool
) -> bool:
    # Check if a valid setting for `max_history` was given
    if isinstance(max_history, int) and max_history < 1:
        raise argparse.ArgumentTypeError(
            f"The value of `--max-history {max_history}` is not a positive integer."
        )

    return validator.verify_story_structure(
        not fail_on_warnings, max_history=max_history
    )


def validate_files(
    fail_on_warnings: bool,
    max_history: Optional[int],
    importer: TrainingDataImporter,
    stories_only: bool = False,
    flows_only: bool = False,
    translations_only: bool = False,
) -> None:
    """Validates either the story structure or the entire project.

    Args:
        fail_on_warnings: `True` if the process should exit with a non-zero status
        max_history: The max history to use when validating the story structure.
        importer: The `TrainingDataImporter` to use to load the training data.
        stories_only: If `True`, only the story structure is validated.
        flows_only: If `True`, only the flows are validated.
        translations_only: If `True`, only the translations data is validated.
    """
    from rasa.validator import Validator

    validator = Validator.from_importer(importer)

    if stories_only:
        all_good = _validate_story_structure(validator, max_history, fail_on_warnings)
    elif flows_only:
        all_good = validator.verify_flows()
    elif translations_only:
        all_good = validator.verify_translations()
    else:
        if importer.get_domain().is_empty():
            structlogger.error(
                "cli.validate_files.empty_domain",
                event_info="Encountered empty domain during validation.",
            )
            display_research_study_prompt()
            raise ValidationError(
                code="cli.validate_files.empty_domain",
                event_info="Encountered empty domain during validation.",
            )

        valid_domain = _validate_domain(validator)
        valid_nlu = _validate_nlu(validator, fail_on_warnings)
        valid_stories = _validate_story_structure(
            validator, max_history, fail_on_warnings
        )
        valid_flows = validator.verify_flows()
        if validator.config:
            valid_translations = validator.verify_translations(summary_mode=True)
        else:
            valid_translations = True
        valid_CALM_slot_mappings = validator.validate_CALM_slot_mappings()

        all_good = (
            valid_domain
            and valid_nlu
            and valid_stories
            and valid_flows
            and valid_translations
            and valid_CALM_slot_mappings
        )

    if validator.config:
        validator.warn_if_config_mandatory_keys_are_not_set()

    telemetry.track_validate_files(all_good)
    if not all_good:
        structlogger.error(
            "cli.validate_files.project_validation_error",
            event_info="Project validation completed with errors.",
        )
        display_research_study_prompt()
        raise ValidationError(
            code="cli.validate_files.project_validation_error",
            event_info="Project validation completed with errors.",
        )
