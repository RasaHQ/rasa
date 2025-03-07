import copy
import typing
from typing import Any, Dict, List, Optional, Text

import numpy as np
from pydantic import BaseModel

from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.dialogue_understanding_test.du_test_case import (
    DialogueUnderstandingTestCase,
    DialogueUnderstandingTestStep,
)
from rasa.dialogue_understanding_test.utils import get_command_comparison
from rasa.shared.nlu.constants import KEY_SYSTEM_PROMPT, KEY_USER_PROMPT

if typing.TYPE_CHECKING:
    from rasa.dialogue_understanding_test.command_metric_calculation import (
        CommandMetrics,
    )

KEY_TEST_CASES_ACCURACY = "test_cases"
KEY_USER_UTTERANCES_ACCURACY = "user_utterances"

OUTPUT_NUMBER_OF_FAILED_TESTS = "number_of_failed_tests"
OUTPUT_NUMBER_OF_PASSED_TESTS = "number_of_passed_tests"
OUTPUT_TEST_CASES_ACCURACY = "test_cases_accuracy"
OUTPUT_USER_UTTERANCES_ACCURACY = "user_utterances_accuracy"
OUTPUT_NUMBER_OF_PASSED_USER_UTTERANCES = "number_of_passed_user_utterances"
OUTPUT_NUMBER_OF_FAILED_USER_UTTERANCES = "number_of_failed_user_utterances"
OUTPUT_COMMAND_METRICS = "command_metrics"
OUTPUT_LATENCY_METRICS = "latency"
OUTPUT_COMPLETION_TOKEN_METRICS = "completion_token"
OUTPUT_PROMPT_TOKEN_METRICS = "prompt_token"
OUTPUT_NAMES_OF_FAILED_TESTS = "names_of_failed_tests"
OUTPUT_NAMES_OF_PASSED_TESTS = "names_of_passed_tests"
OUTPUT_LLM_COMMAND_GENERATOR_CONFIG = "llm_command_generator_config"


class DialogueUnderstandingTestResult(BaseModel):
    """Result of a single dialogue understanding test case."""

    test_case: DialogueUnderstandingTestCase
    passed: bool
    error_line: Optional[int] = None

    def get_expected_commands(self) -> List[PromptCommand]:
        return self.test_case.get_expected_commands()


class FailedTestStep(BaseModel):
    """Failed test step information."""

    file: str
    test_case_name: str
    failed_user_utterance: str
    error_line: int
    pass_status: bool
    command_generators: List[str]
    prompts: Optional[Dict[str, List[Dict[str, Any]]]] = None
    expected_commands: List[PromptCommand]
    predicted_commands: Dict[str, List[PromptCommand]]
    conversation_with_diff: List[str]

    class Config:
        """Skip validation for PromptCommand protocol as pydantic does not know how to
        serialize or handle instances of a protocol.
        """

        arbitrary_types_allowed = True

    @classmethod
    def from_dialogue_understanding_test_step(
        cls,
        step: DialogueUnderstandingTestStep,
        test_case: DialogueUnderstandingTestCase,
    ) -> "FailedTestStep":
        file_path = test_case.file or ""
        user_utterance = step.text or ""
        line_number = step.line or -1

        predicted_commands: Dict[str, List[PromptCommand]] = {}
        prompts: Optional[Dict[str, List[Dict[str, Any]]]] = None
        command_generators: List[str] = []

        if step.dialogue_understanding_output:
            predicted_commands = step.dialogue_understanding_output.commands
            command_generators = step.dialogue_understanding_output.get_component_names_that_predicted_commands_or_have_llm_response()  # noqa: E501
            prompts = (
                step.dialogue_understanding_output.get_component_name_to_prompt_info()
            )

        step_index = test_case.steps.index(step)

        conversation_with_diff = test_case.to_readable_conversation(
            until_step=step_index + 1
        ) + get_command_comparison(step)

        return cls(
            file=file_path,
            test_case_name=test_case.name,
            failed_user_utterance=user_utterance,
            error_line=line_number,
            pass_status=False,
            command_generators=command_generators,
            prompts=prompts,
            expected_commands=step.commands or [],
            predicted_commands=predicted_commands,
            conversation_with_diff=conversation_with_diff,
        )

    def to_dict(self, output_prompt: bool) -> Dict[str, Any]:
        step_info = {
            "file": self.file,
            "test_case": self.test_case_name,
            "failed_user_utterance": self.failed_user_utterance,
            "error_line": self.error_line,
            "pass_status": self.pass_status,
            "expected_commands": [
                command.to_dsl() for command in self.expected_commands
            ],
            "predicted_commands": [
                {
                    "component": component,
                    "commands": [command.to_dsl() for command in commands],
                }
                for component, commands in self.predicted_commands.items()
                if commands
            ],
        }

        if output_prompt and self.prompts:
            step_info["prompts"] = copy.deepcopy(self.prompts)
        elif self.prompts:
            prompts = copy.deepcopy(self.prompts)
            # remove user and system prompts
            for prompt_data in prompts.values():
                for prompt_info in prompt_data:
                    prompt_info.pop(KEY_USER_PROMPT, None)
                    prompt_info.pop(KEY_SYSTEM_PROMPT, None)

                step_info["prompts"] = prompts

        return step_info


class DialogueUnderstandingTestSuiteResult:
    """Result of a dialogue understanding test suite.

    Aggregates test results and provides metrics for the entire test suite
    used to log the results to the console and write them to a file.
    """

    def __init__(self) -> None:
        self.accuracy = {
            KEY_TEST_CASES_ACCURACY: 0.0,
            KEY_USER_UTTERANCES_ACCURACY: 0.0,
        }
        self.number_of_passed_tests = 0
        self.number_of_failed_tests = 0
        self.number_of_passed_user_utterances = 0
        self.number_of_failed_user_utterances = 0
        self.command_metrics: Optional[Dict[str, "CommandMetrics"]] = None
        self.names_of_failed_tests: List[str] = []
        self.names_of_passed_tests: List[str] = []
        self.failed_test_steps: List[FailedTestStep] = []
        self.llm_config: Optional[Dict[str, Any]] = None
        self.latency_metrics: Dict[str, float] = {}
        self.prompt_token_metrics: Dict[str, float] = {}
        self.completion_token_metrics: Dict[str, float] = {}

    @classmethod
    def from_results(
        cls,
        failing_test_results: List[DialogueUnderstandingTestResult],
        passing_test_results: List[DialogueUnderstandingTestResult],
        command_metrics: Dict[str, "CommandMetrics"],
        llm_config: Optional[Dict[str, Any]],
    ) -> "DialogueUnderstandingTestSuiteResult":
        """Create a DialogueUnderstandingTestSuiteResult object from the test results.

        Create a new instance of DialogueUnderstandingTestSuiteResult by aggregating
        metrics from passing and failing test results, along with command metrics.

        Args:
            failing_test_results: A list of DialogueUnderstandingTestResult objects
                representing the test cases that did not pass.
            passing_test_results: A list of DialogueUnderstandingTestResult objects
                representing the test cases that passed.
            command_metrics: A dictionary of command-specific performance metrics, keyed
                by command name.
            llm_config: A dictionary containing the command generator configuration.

        Returns:
            A DialogueUnderstandingTestSuiteResult object containing aggregated test
            suite metrics, including accuracy, counts of passed and failed test cases,
            user utterance statistics, and command metrics.
        """
        instance = cls()

        instance.number_of_passed_tests = len(passing_test_results)
        instance.number_of_failed_tests = len(failing_test_results)
        instance.accuracy[KEY_TEST_CASES_ACCURACY] = instance.number_of_passed_tests / (
            instance.number_of_passed_tests + instance.number_of_failed_tests
        )

        instance._set_user_utterance_metrics(failing_test_results, passing_test_results)

        instance.command_metrics = command_metrics

        instance.names_of_passed_tests = [
            passing_test_result.test_case.full_name()
            for passing_test_result in passing_test_results
        ]
        instance.names_of_failed_tests = [
            failing_test_result.test_case.full_name()
            for failing_test_result in failing_test_results
        ]

        instance.failed_test_steps = cls._create_failed_steps_from_results(
            failing_test_results
        )

        instance.latency_metrics = cls.get_latency_metrics(
            failing_test_results, passing_test_results
        )
        instance.prompt_token_metrics = cls.get_prompt_token_metrics(
            failing_test_results, passing_test_results
        )
        instance.completion_token_metrics = cls.get_completion_token_metrics(
            failing_test_results, passing_test_results
        )

        instance.llm_config = llm_config

        return instance

    def _set_user_utterance_metrics(
        self,
        failing_test_results: List[DialogueUnderstandingTestResult],
        passing_test_results: List[DialogueUnderstandingTestResult],
    ) -> None:
        # Create list of booleans indicating whether each user utterance
        # passed or failed
        user_utterances_status = [
            step.has_passed()
            for test in failing_test_results + passing_test_results
            for step in test.test_case.iterate_over_user_steps()
        ]
        # Calculate number of passed and failed user utterances
        self.number_of_passed_user_utterances = sum(user_utterances_status)
        self.number_of_failed_user_utterances = (
            len(user_utterances_status) - self.number_of_passed_user_utterances
        )
        # Calculate user utterance accuracy
        self.accuracy[KEY_USER_UTTERANCES_ACCURACY] = (
            self.number_of_passed_user_utterances
            / (
                self.number_of_failed_user_utterances
                + self.number_of_passed_user_utterances
            )
        )

    @staticmethod
    def _create_failed_steps_from_results(
        failing_test_results: List["DialogueUnderstandingTestResult"],
    ) -> List[FailedTestStep]:
        """Create list of FailedTestStep objects from failing test results.

        Given a list of failing DialogueUnderstandingTestResult objects,
        create and return a list of FailedTestStep objects for each failing user step.

        Args:
            failing_test_results: Results of failing Dialogue Understanding tests.

        Returns:
            List of aggregated FailedTestStep objects for logging to console and file.
        """
        failed_test_steps: List[FailedTestStep] = []

        for result in failing_test_results:
            test_case = result.test_case
            for step in test_case.failed_user_steps():
                failed_test_steps.append(
                    FailedTestStep.from_dialogue_understanding_test_step(
                        step, test_case
                    )
                )

        return failed_test_steps

    @staticmethod
    def _calculate_percentiles(values: List[float]) -> Dict[str, float]:
        return {
            "p50": float(np.percentile(values, 50)) if values else 0.0,
            "p90": float(np.percentile(values, 90)) if values else 0.0,
            "p99": float(np.percentile(values, 99)) if values else 0.0,
        }

    @classmethod
    def get_latency_metrics(
        cls,
        failing_test_results: List["DialogueUnderstandingTestResult"],
        passing_test_results: List["DialogueUnderstandingTestResult"],
    ) -> Dict[str, float]:
        latencies = [
            latency
            for result in failing_test_results + passing_test_results
            for step in result.test_case.steps
            for latency in step.get_latencies()
        ]

        return cls._calculate_percentiles(latencies)

    @classmethod
    def get_prompt_token_metrics(
        cls,
        failing_test_results: List["DialogueUnderstandingTestResult"],
        passing_test_results: List["DialogueUnderstandingTestResult"],
    ) -> Dict[str, float]:
        tokens = [
            token_count
            for result in failing_test_results + passing_test_results
            for step in result.test_case.steps
            for token_count in step.get_prompt_tokens()
        ]

        return cls._calculate_percentiles(tokens)

    @classmethod
    def get_completion_token_metrics(
        cls,
        failing_test_results: List["DialogueUnderstandingTestResult"],
        passing_test_results: List["DialogueUnderstandingTestResult"],
    ) -> Dict[str, float]:
        tokens = [
            token_count
            for result in failing_test_results + passing_test_results
            for step in result.test_case.steps
            for token_count in step.get_completion_tokens()
        ]

        return cls._calculate_percentiles(tokens)

    def to_dict(self, output_prompt: bool = False) -> Dict[str, Any]:
        """Builds a dictionary for writing test results to a YML file.

        Args:
            output_prompt: Whether to log the prompt or not.
        """
        # 1. Accuracy block
        result_dict: Dict[Text, Any] = {
            "accuracy": {
                "test_cases": self.accuracy[KEY_TEST_CASES_ACCURACY],
                "user_utterances": self.accuracy[KEY_USER_UTTERANCES_ACCURACY],
            },
            OUTPUT_NUMBER_OF_PASSED_TESTS: self.number_of_passed_tests,
            OUTPUT_NUMBER_OF_FAILED_TESTS: self.number_of_failed_tests,
            OUTPUT_NUMBER_OF_PASSED_USER_UTTERANCES: self.number_of_passed_user_utterances,  # noqa: E501
            OUTPUT_NUMBER_OF_FAILED_USER_UTTERANCES: self.number_of_failed_user_utterances,  # noqa: E501
        }

        cmd_metrics_output = {}
        if self.command_metrics:
            if isinstance(self.command_metrics, dict):
                for cmd_name, metrics_obj in self.command_metrics.items():
                    cmd_metrics_output[cmd_name] = metrics_obj.as_dict()
            else:
                pass

        result_dict[OUTPUT_COMMAND_METRICS] = cmd_metrics_output

        result_dict[OUTPUT_LATENCY_METRICS] = self.latency_metrics
        result_dict[OUTPUT_PROMPT_TOKEN_METRICS] = self.prompt_token_metrics
        result_dict[OUTPUT_COMPLETION_TOKEN_METRICS] = self.completion_token_metrics

        result_dict[OUTPUT_NAMES_OF_PASSED_TESTS] = self.names_of_passed_tests
        result_dict[OUTPUT_NAMES_OF_FAILED_TESTS] = self.names_of_failed_tests

        failed_steps_list = []
        for failed_test_step in self.failed_test_steps:
            failed_steps_list.append(
                failed_test_step.to_dict(output_prompt=output_prompt)
            )

        result_dict["failed_test_steps"] = failed_steps_list

        if self.llm_config:
            result_dict[OUTPUT_LLM_COMMAND_GENERATOR_CONFIG] = self.llm_config

        return result_dict
