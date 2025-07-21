import argparse
from typing import List, Optional

import structlog

from rasa.dialogue_understanding_test.constants import (
    ACTOR_BOT,
    PLACEHOLDER_GENERATED_ANSWER_TEMPLATE,
)
from rasa.dialogue_understanding_test.du_test_case import DialogueUnderstandingTestCase
from rasa.exceptions import ValidationError
from rasa.shared.core.domain import Domain

structlogger = structlog.get_logger()


def validate_cli_arguments(args: argparse.Namespace) -> None:
    """Validate the CLI arguments for the dialogue understanding test.

    Args:
        args: Commandline arguments.
    """
    # Model path, endpoints file path, and the path to test cases are validated
    # in other places.
    # Validate remote storage option
    supported_remote_storages = ["aws", "gcs", "azure"]
    if (
        args.remote_storage
        and args.remote_storage.lower() not in supported_remote_storages
    ):
        raise ValidationError(
            code="dialogue_understanding_test.validate_cli_arguments"
            ".invalid_remote_storage",
            event_info=(
                f"Invalid remote storage option - '{args.remote_storage}'. Supported "
                f"options are: {supported_remote_storages}"
            ),
        )


def validate_test_cases(
    test_cases: List[DialogueUnderstandingTestCase], domain: Optional[Domain]
) -> None:
    """Validate the dialogue understanding test cases.

    Args:
        test_cases: Test cases to validate.
        domain: Domain of the assistant.
    """
    if not domain:
        raise ValidationError(
            code="dialogue_understanding_test.validate_test_cases.no_domain",
            event_info="No domain found. Retrain the model with a valid domain.",
        )

    # Retrieve all valid templates from the domain
    valid_templates = domain.utterances_for_response

    # Add 'placeholder_generated_answer' as a valid template
    valid_templates.add(PLACEHOLDER_GENERATED_ANSWER_TEMPLATE)

    for test_case in test_cases:
        for step in test_case.steps:
            if step.actor == ACTOR_BOT and step.template:
                if step.template not in valid_templates:
                    raise ValidationError(
                        code="dialogue_understanding_test.validate_test_cases"
                        ".invalid_template",
                        event_info=(
                            f"Invalid bot utterance template '{step.template}' in test "
                            f"case '{test_case.name}' at line {step.line}. Please "
                            f"the template exists."
                        ),
                        test_case=test_case.name,
                        template=step.template,
                    )
