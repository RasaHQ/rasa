from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

import rich

import rasa.shared.data
from rasa.dialogue_understanding_test.command_metrics import CommandMetrics
from rasa.dialogue_understanding_test.constants import SCHEMA_FILE_PATH
from rasa.dialogue_understanding_test.du_test_case import (
    KEY_CHOICES,
    KEY_COMPLETION_TOKENS,
    KEY_PROMPT_TOKENS,
)
from rasa.dialogue_understanding_test.du_test_result import (
    KEY_COMMANDS_F1_MACRO,
    KEY_COMMANDS_F1_MICRO,
    KEY_COMMANDS_F1_WEIGHTED,
    DialogueUnderstandingTestSuiteResult,
    FailedTestStep,
)
from rasa.e2e_test.constants import KEY_TEST_CASE, KEY_TEST_CASES
from rasa.e2e_test.e2e_test_case import (
    DialogueUnderstandingTestCase,
    Fixture,
    Metadata,
    TestSuite,
)
from rasa.e2e_test.stub_custom_action import StubCustomAction
from rasa.e2e_test.utils.io import (
    check_beta_feature_flag_for_custom_actions_stubs,
    extract_fixtures,
    extract_metadata,
    extract_stub_custom_actions,
    extract_test_case_from_path,
    is_test_case_file,
    parse_raw_yaml,
    validate_path_to_test_cases,
    validate_test_case,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.nlu.constants import (
    KEY_LATENCY,
    KEY_PROMPT_NAME,
    KEY_SYSTEM_PROMPT,
    KEY_USER_PROMPT,
)
from rasa.shared.utils.yaml import (
    read_schema_file,
    validate_yaml_content_using_schema,
    write_yaml,
)

if TYPE_CHECKING:
    from rasa.dialogue_understanding.commands import Command


def read_du_test_schema() -> Union[List[Any], Dict[str, Any]]:
    """Read the schema for the dialogue understanding test cases.

    Returns:
        Dict[str, Any]: Schema for the dialogue understanding test cases.
    """
    return read_schema_file(SCHEMA_FILE_PATH)


def read_test_suite(
    test_case_path: str,
    flows: FlowsList,
    custom_command_classes: List["Command"] = [],
    remove_default_commands: List[str] = [],
) -> TestSuite:
    """Read the test cases from the given test case path.

    Args:
        test_case_path: Path to the test cases.
        flows: List of flows.
        custom_command_classes: Custom command classes to use in the test cases.
        remove_default_commands: Default commands to remove from the test cases.

    Returns:
        TestSuite: Test suite containing the dialogue understanding test cases.
    """

    def _extract_test_cases(
        test_file_content: dict,
        test_case_name: str,
        test_file: str,
        flows: FlowsList,
        custom_command_classes: List["Command"] = [],
        remove_default_commands: List[str] = [],
    ) -> List[DialogueUnderstandingTestCase]:
        """Extract test cases from the test file content.

        Args:
            test_file_content: Content of the test file.
            test_case_name: Name of the test case to extract.
            test_file: Path to the test file.
            flows: List of flows.

        Returns:
            List of test cases.
        """
        test_cases_content = test_file_content.get(KEY_TEST_CASES) or []
        if test_case_name:
            return [
                DialogueUnderstandingTestCase.from_dict(
                    test_case_dict,
                    flows,
                    file=test_file,
                    custom_command_classes=custom_command_classes,
                    remove_default_commands=remove_default_commands,
                )
                for test_case_dict in test_cases_content
                if test_case_name == test_case_dict.get(KEY_TEST_CASE)
            ]
        return [
            DialogueUnderstandingTestCase.from_dict(
                test_case_dict,
                flows,
                file=test_file,
                custom_command_classes=custom_command_classes,
                remove_default_commands=remove_default_commands,
            )
            for test_case_dict in test_cases_content
        ]

    # Extract test case path and name
    test_case_path, test_case_name = extract_test_case_from_path(test_case_path)
    validate_path_to_test_cases(test_case_path)

    # Load test files and schema
    test_files = rasa.shared.data.get_data_files([test_case_path], is_test_case_file)
    test_schema = read_du_test_schema()

    # Initialize containers
    input_test_cases = []
    fixtures: Dict[str, Fixture] = {}
    metadata: Dict[str, Metadata] = {}
    stub_custom_actions: Dict[str, StubCustomAction] = {}

    # Process each test file
    for test_file in test_files:
        test_file_content = parse_raw_yaml(Path(test_file).read_text(encoding="utf-8"))

        # Validate YAML content using the provided function
        validate_yaml_content_using_schema(test_file_content, test_schema)

        # Parse test cases, fixtures, metadata, and stub custom actions
        test_cases = _extract_test_cases(
            test_file_content,
            test_case_name,
            test_file,
            flows,
            custom_command_classes,
            remove_default_commands,
        )
        fixtures.update(extract_fixtures(test_file_content, fixtures))
        metadata.update(extract_metadata(test_file_content, metadata))
        stub_custom_actions.update(
            extract_stub_custom_actions(test_file_content, test_file)
        )
        input_test_cases.extend(test_cases)

    validate_test_case(test_case_name, input_test_cases, fixtures, metadata)
    if stub_custom_actions:
        check_beta_feature_flag_for_custom_actions_stubs()

    return TestSuite(
        input_test_cases,
        list(fixtures.values()),
        list(metadata.values()),
        stub_custom_actions,
    )


def write_test_results_to_file(
    test_suite_result: DialogueUnderstandingTestSuiteResult,
    output_file: str,
    output_prompt: bool,
) -> None:
    """Write the test results to the given output file.

    Args:
        test_suite_result: Test results suite containing the test results.
        output_file: Path to the output file.
        output_prompt: Whether to log the prompt or not.
    """
    write_yaml(
        test_suite_result.to_dict(output_prompt),
        target=output_file,
        should_preserve_key_order=True,
    )

    print(f"Results written to '{output_file}'.")


def print_test_results(
    test_suite_result: DialogueUnderstandingTestSuiteResult,
    output_prompt: bool,
) -> None:
    """Print the result of the test run.

    Example output (truncated for brevity):
        ====== FAILURES ======

        ---------------- test_case: rasa-calm-demo/dialogue_understanding_tests/
        immediate_cancellation_and_start_of_new_flow.yml::user immediately cancels
        and starts new flow -----------------

        == failure starting at user message 'I want to send money'.

        -- COMMAND GENERATOR(s) --
        SingleStepLLMCommandGenerator

        -- CONVERSATION --
        user: I want to send money
        ---EXPECTED---                               | ---PREDICTED---
        StartFlow(transfer_money)                    | StartFlow(transfer_money)
        SetSlot(transfer_money_amount_of_money, 878) |

        ...
        ====== COMMAND METRICS ======
        set slot (2 commands in total):
          tp: 0 fp: 1 fn: 2
          precision: 0.0000
          recall   : 0.0000
          f1       : 0.0000

        cancel flow (1 commands in total):
          tp: 1 fp: 0 fn: 0
          precision: 1.0000
          recall   : 1.0000
          f1       : 1.0000

        ...
        ====== LATENCY METRICS ======
        p50: 0.00065571
        p90: 0.00074687
        p99: 0.00077837
        ====== PROMPT TOKEN METRICS ======
        p50: 1336.00
        p90: 1389.50
        p99: 1401.65
        ====== COMPLETION TOKEN METRICS ======
        p50: 12.00
        p90: 15.80
        p99: 16.88

        ====== 1 failed test cases, 0 passed test cases ======
        ====== 2 failed user steps, 1 passed user steps (accuracy: 0.3333) ======

    Args:
        test_suite_result: Test results suite containing the test results.
        output_prompt: Whether to log the prompt or not.
    """
    if (
        test_suite_result.number_of_passed_tests
        + test_suite_result.number_of_failed_tests
        == 0
    ):
        # no tests were run, print error
        rasa.shared.utils.cli.print_error(
            rasa.shared.utils.cli.pad("No test cases found.", char="!")
        )
        return

    if test_suite_result.number_of_failed_tests > 0:
        # print failure headline
        print()
        rich.print(
            f"[bold][red3]"
            f"{rasa.shared.utils.cli.pad('FAILURES', char='=')}"
            f"[/red3][/bold]"
        )

        # print failed test steps
        print_failed_cases(test_suite_result, output_prompt=output_prompt)

    print_f1_summary(test_suite_result)
    print_command_summary(test_suite_result.command_metrics)
    print_latency_and_token_metrics(test_suite_result)
    print_final_line(test_suite_result)


def print_failed_cases(
    test_suite_result: DialogueUnderstandingTestSuiteResult,
    output_prompt: bool,
) -> None:
    """Print the details of a failed test case."""
    # Group the failed test steps by test case
    step_groups = defaultdict(list)
    for step in test_suite_result.failed_test_steps:
        key = f"{step.file}::{step.test_case_name}"
        step_groups[key].append(step)

    for failed_test_case, failed_test_steps in step_groups.items():
        fail_headline = f"test_case: {failed_test_case}"
        print()
        rasa.shared.utils.cli.print_error(
            f"{rasa.shared.utils.cli.pad(fail_headline, char='-')}\n"
        )
        print(f"Number of failed steps: {len(failed_test_steps)}")
        for step in failed_test_steps:
            print()
            rasa.shared.utils.cli.print_info(
                f"== failure starting at user message "
                f"'{step.failed_user_utterance}'. "
            )

            rich.print("\n[red3]-- COMMAND GENERATOR(s) --[/red3]")
            rich.print("\n".join(step.command_generators))
            if output_prompt:
                print_prompt(step)
            rich.print("\n[red3]-- CONVERSATION --[/red3]")
            rich.print("\n".join(step.conversation_with_diff))
            print_llm_output(step)


def print_prompt(step: FailedTestStep) -> None:
    if step.prompts is None:
        return
    prompts = step.prompts

    rich.print("\n[red3]-- PROMPT(s) --[/red3]")
    for component, component_prompts in prompts.items():
        rich.print(f"[bold]{component}[/bold]")
        for prompt_data in component_prompts:
            rich.print(
                f"[bold]  prompt name      [/bold]: {prompt_data[KEY_PROMPT_NAME]}"
            )
            rich.print(
                f"[bold]  prompt tokens    [/bold]: {prompt_data[KEY_PROMPT_TOKENS]}"
            )
            rich.print(
                f"[bold]  completion tokens[/bold]: "
                f"{prompt_data[KEY_COMPLETION_TOKENS]}"
            )
            rich.print(f"[bold]  latency          [/bold]: {prompt_data[KEY_LATENCY]}")
            if KEY_SYSTEM_PROMPT in prompt_data:
                rich.print(
                    f"[bold]  system prompt    [/bold]: "
                    f"{prompt_data[KEY_SYSTEM_PROMPT]}"
                )
            rich.print(
                f"[bold]  user prompt      [/bold]: {prompt_data[KEY_USER_PROMPT]}"
            )


def print_llm_output(step: FailedTestStep) -> None:
    if not step.prompts:
        return

    for component, component_prompts in step.prompts.items():
        for prompt_data in component_prompts:
            if KEY_CHOICES in prompt_data:
                rich.print(f"\n[red3]-- LLM ouptut for {component} --[/red3]")
                rich.print(prompt_data.get(KEY_CHOICES))
                rich.print("[red3]-------------[/red3]")


def print_f1_summary(result: DialogueUnderstandingTestSuiteResult) -> None:
    """Print the f1 summary."""
    print()
    rasa.shared.utils.cli.print_info(rasa.shared.utils.cli.pad("COMMANDS F1"))
    rasa.shared.utils.cli.print_info(
        f"macro           : {result.f1_score[KEY_COMMANDS_F1_MACRO]:.8f}"
    )
    rasa.shared.utils.cli.print_info(
        f"micro           : {result.f1_score[KEY_COMMANDS_F1_MICRO]:.8f}"
    )
    rasa.shared.utils.cli.print_info(
        f"weighted average: {result.f1_score[KEY_COMMANDS_F1_WEIGHTED]:.8f}"
    )


def print_command_summary(metrics: Dict[str, CommandMetrics]) -> None:
    """Print the command summary.

    Args:
        metrics: Dict of command to precision, recall, f1, etc. scores
    """
    print()
    rasa.shared.utils.cli.print_info(rasa.shared.utils.cli.pad("COMMAND METRICS"))

    for command_name, command_metric in metrics.items():
        rasa.shared.utils.cli.print_info(
            f"{command_name} " f"({command_metric.total_count} commands in total):"
        )
        rasa.shared.utils.cli.print_info(
            f"  tp: {command_metric.tp} "
            f"fp: {command_metric.fp} "
            f"fn: {command_metric.fn}"
        )
        rasa.shared.utils.cli.print_info(
            f"  precision: {command_metric.get_precision():.4f}"
        )
        rasa.shared.utils.cli.print_info(
            f"  recall   : {command_metric.get_recall():.4f}"
        )
        rasa.shared.utils.cli.print_info(
            f"  f1       : {command_metric.get_f1_score():.4f}"
        )


def print_latency_and_token_metrics(
    result: DialogueUnderstandingTestSuiteResult,
) -> None:
    """Print the latency and token metrics."""
    print()
    rasa.shared.utils.cli.print_info(rasa.shared.utils.cli.pad("LATENCY METRICS"))
    for component, latency_metric in result.latency_metrics.items():
        rasa.shared.utils.cli.print_info(f"--- {component} ---")
        for key, value in latency_metric.items():
            rasa.shared.utils.cli.print_info(f"{key}: {value:.8f}")

    rasa.shared.utils.cli.print_info(rasa.shared.utils.cli.pad("PROMPT TOKEN METRICS"))
    for component, prompt_token_metric in result.prompt_token_metrics.items():
        rasa.shared.utils.cli.print_info(f"--- {component} ---")
        for key, value in prompt_token_metric.items():
            rasa.shared.utils.cli.print_info(f"{key}: {value:.2f}")

    rasa.shared.utils.cli.print_info(
        rasa.shared.utils.cli.pad("COMPLETION TOKEN METRICS")
    )
    for component, completion_token_metric in result.completion_token_metrics.items():
        rasa.shared.utils.cli.print_info(f"--- {component} ---")
        for key, value in completion_token_metric.items():
            rasa.shared.utils.cli.print_info(f"{key}: {value:.2f}")


def print_final_line(test_suite_result: DialogueUnderstandingTestSuiteResult) -> None:
    """Print the final line of the test output.

    Args:
        test_suite_result: Test suite result.
    """
    has_failed = test_suite_result.number_of_failed_tests > 0

    final_line_color = "green3" if not has_failed else "red3"

    def _print_accuracy_line(number_passed: int, number_failed: int, type: str) -> None:
        if number_passed > 0:
            line = rasa.shared.utils.cli.pad(
                f"[bold red3]{number_failed} failed {type}[/bold red3]"
                f"[bright_white], [/bright_white]"
                f"[bold green3]{number_passed} passed {type}[/bold green3]"
                f"[bright_white] "
                f"(accuracy:  {number_passed / (number_passed + number_failed):.4f})"
                f"[/bright_white]"
            )
        else:
            line = rasa.shared.utils.cli.pad(
                f"[bold red3]{number_failed} failed {type}[/bold red3]"
                f"[bright_white], [/bright_white]"
                f"[bold green3]{number_passed} passed {type}[/bold green3]"
            )
        rich.print(f"[{final_line_color}]{line}[/{final_line_color}]")

    _print_accuracy_line(
        test_suite_result.number_of_passed_tests,
        test_suite_result.number_of_failed_tests,
        "test cases",
    )

    _print_accuracy_line(
        test_suite_result.number_of_passed_user_utterances,
        test_suite_result.number_of_failed_user_utterances,
        "user steps",
    )
