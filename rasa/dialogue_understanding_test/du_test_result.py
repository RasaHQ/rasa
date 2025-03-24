import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Text

import numpy as np
from pydantic import BaseModel

from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.dialogue_understanding_test.command_metrics import (
    CommandMetrics,
)
from rasa.dialogue_understanding_test.du_test_case import (
    DialogueUnderstandingTestCase,
    DialogueUnderstandingTestStep,
)
from rasa.dialogue_understanding_test.utils import get_command_comparison
from rasa.shared.nlu.constants import KEY_SYSTEM_PROMPT, KEY_USER_PROMPT

KEY_TEST_CASES_ACCURACY = "test_cases"
KEY_USER_UTTERANCES_ACCURACY = "user_utterances"

KEY_COMMANDS_F1_MACRO = "macro"
KEY_COMMANDS_F1_MICRO = "micro"
KEY_COMMANDS_F1_WEIGHTED = "weighted_average"

OUTPUT_DUT_ACCURACY = "accuracy"
OUTPUT_DUT_ACCURACY_TEST_CASES = "test_cases"
OUTPUT_DUT_ACCURACY_USER_UTTERANCES = "user_utterances"

OUTPUT_COMMANDS_F1 = "f1_score"
OUTPUT_COMMANDS_F1_MACRO = "macro"
OUTPUT_COMMANDS_F1_MICRO = "micro"
OUTPUT_COMMANDS_F1_WEIGHTED = "weighted_average"

OUTPUT_NUMBER_OF_FAILED_TESTS = "number_of_failed_tests"
OUTPUT_NUMBER_OF_PASSED_TESTS = "number_of_passed_tests"
OUTPUT_NUMBER_OF_PASSED_USER_UTTERANCES = "number_of_passed_user_utterances"
OUTPUT_NUMBER_OF_FAILED_USER_UTTERANCES = "number_of_failed_user_utterances"
OUTPUT_NAMES_OF_FAILED_TESTS = "names_of_failed_tests"
OUTPUT_NAMES_OF_PASSED_TESTS = "names_of_passed_tests"
OUTPUT_FAILED_TEST_STEPS = "failed_test_steps"
OUTPUT_TEST_CASES_ACCURACY = "test_cases_accuracy"
OUTPUT_USER_UTTERANCES_ACCURACY = "user_utterances_accuracy"
OUTPUT_COMMAND_METRICS = "command_metrics"
OUTPUT_COMMANDS_F1_MACRO_INSTRUMENTATION_ATTR = "commands_f1_macro"
OUTPUT_COMMANDS_F1_MICRO_INSTRUMENTATION_ATTR = "commands_f1_micro"
OUTPUT_COMMANDS_F1_WEIGHTED_INSTRUMENTATION_ATTR = "commands_f1_weighted_average"

OUTPUT_LATENCY_METRICS = "latency"
OUTPUT_COMPLETION_TOKEN_METRICS = "completion_token"
OUTPUT_PROMPT_TOKEN_METRICS = "prompt_token"

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
    conversation_until_failed_user_utterance: List[str]

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
        conversation_until_failed_user_utterance = test_case.to_readable_conversation(
            until_step=step_index + 1
        )
        conversation_with_diff = (
            conversation_until_failed_user_utterance + get_command_comparison(step)
        )

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
            conversation_until_failed_user_utterance=conversation_until_failed_user_utterance,
        )

    def to_dict(self, output_prompt: bool) -> Dict[str, Any]:
        step_info = {
            "file": self.file,
            "test_case": self.test_case_name,
            "conversation": self.conversation_until_failed_user_utterance,
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
        self.f1_score = {
            KEY_COMMANDS_F1_MACRO: 0.0,
            KEY_COMMANDS_F1_MICRO: 0.0,
            KEY_COMMANDS_F1_WEIGHTED: 0.0,
        }
        self.number_of_passed_tests = 0
        self.number_of_failed_tests = 0
        self.number_of_passed_user_utterances = 0
        self.number_of_failed_user_utterances = 0
        self.command_metrics: Optional[Dict[str, CommandMetrics]] = None
        self.names_of_failed_tests: List[str] = []
        self.names_of_passed_tests: List[str] = []
        self.failed_test_steps: List[FailedTestStep] = []
        self.llm_config: Optional[Dict[str, Any]] = None
        # The performance metrics distribution per component
        # For example: {"command_generator": {"p50": x, ...}, ...}
        self.latency_metrics: Dict[str, Dict[str, float]] = {}
        self.prompt_token_metrics: Dict[str, Dict[str, float]] = {}
        self.completion_token_metrics: Dict[str, Dict[str, float]] = {}

    @classmethod
    def from_results(
        cls,
        failing_test_results: List[DialogueUnderstandingTestResult],
        passing_test_results: List[DialogueUnderstandingTestResult],
        command_metrics: Dict[str, CommandMetrics],
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

        instance.f1_score[KEY_COMMANDS_F1_MACRO] = cls.calculate_f1_macro(
            command_metrics
        )
        instance.f1_score[KEY_COMMANDS_F1_MICRO] = cls.calculate_f1_micro(
            command_metrics
        )
        instance.f1_score[KEY_COMMANDS_F1_WEIGHTED] = cls.calculate_f1_weighted(
            command_metrics
        )

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

    def to_dict(self, output_prompt: bool = False) -> Dict[str, Any]:
        """Builds a dictionary for writing test results to a YML file.

        Args:
            output_prompt: Whether to log the prompt or not.
        """
        result_dict: Dict[Text, Any] = {
            # Accuracy block
            OUTPUT_DUT_ACCURACY: {
                OUTPUT_DUT_ACCURACY_TEST_CASES: self.accuracy[KEY_TEST_CASES_ACCURACY],
                OUTPUT_DUT_ACCURACY_USER_UTTERANCES: self.accuracy[
                    KEY_USER_UTTERANCES_ACCURACY
                ],
            },
            # F1 block
            OUTPUT_COMMANDS_F1: {
                OUTPUT_COMMANDS_F1_MACRO: self.f1_score[KEY_COMMANDS_F1_MACRO],
                OUTPUT_COMMANDS_F1_MICRO: self.f1_score[KEY_COMMANDS_F1_MICRO],
                OUTPUT_COMMANDS_F1_WEIGHTED: self.f1_score[KEY_COMMANDS_F1_WEIGHTED],
            },
            # Other metrics block
            OUTPUT_NUMBER_OF_PASSED_TESTS: self.number_of_passed_tests,
            OUTPUT_NUMBER_OF_FAILED_TESTS: self.number_of_failed_tests,
            OUTPUT_NUMBER_OF_PASSED_USER_UTTERANCES: self.number_of_passed_user_utterances,  # noqa: E501
            OUTPUT_NUMBER_OF_FAILED_USER_UTTERANCES: self.number_of_failed_user_utterances,  # noqa: E501
        }

        # Command metrics block
        cmd_metrics_output = {}
        if self.command_metrics:
            if isinstance(self.command_metrics, dict):
                for cmd_name, metrics_obj in self.command_metrics.items():
                    cmd_metrics_output[cmd_name] = metrics_obj.as_dict()
            else:
                pass
        result_dict[OUTPUT_COMMAND_METRICS] = cmd_metrics_output

        # Latency and tokens metrics block
        result_dict[OUTPUT_LATENCY_METRICS] = self.latency_metrics
        result_dict[OUTPUT_PROMPT_TOKEN_METRICS] = self.prompt_token_metrics
        result_dict[OUTPUT_COMPLETION_TOKEN_METRICS] = self.completion_token_metrics

        # Passed and failed test names block
        result_dict[OUTPUT_NAMES_OF_PASSED_TESTS] = self.names_of_passed_tests
        result_dict[OUTPUT_NAMES_OF_FAILED_TESTS] = self.names_of_failed_tests

        # Failed test steps block
        failed_steps_list = []
        for failed_test_step in self.failed_test_steps:
            failed_steps_list.append(
                failed_test_step.to_dict(output_prompt=output_prompt)
            )
        result_dict[OUTPUT_FAILED_TEST_STEPS] = failed_steps_list

        # LLM config block
        if self.llm_config:
            result_dict[OUTPUT_LLM_COMMAND_GENERATOR_CONFIG] = self.llm_config

        return result_dict

    @staticmethod
    def calculate_f1_macro(command_metrics: Dict[str, CommandMetrics]) -> float:
        f1_scores = [metrics.get_f1_score() for metrics in command_metrics.values()]
        return sum(f1_scores) / len(f1_scores)

    @staticmethod
    def calculate_f1_micro(command_metrics: Dict[str, CommandMetrics]) -> float:
        combined_metrics = CommandMetrics(
            tp=sum([metrics.tp for metrics in command_metrics.values()]),
            fp=sum([metrics.fp for metrics in command_metrics.values()]),
            fn=sum([metrics.fn for metrics in command_metrics.values()]),
            total_count=sum(m.total_count for m in command_metrics.values()),
        )
        return combined_metrics.get_f1_score()

    @staticmethod
    def calculate_f1_weighted(command_metrics: Dict[str, CommandMetrics]) -> float:
        class_counts = []
        f1_scores = []
        for metrics in command_metrics.values():
            class_counts.append(metrics.total_count)
            f1_scores.append(metrics.get_f1_score())

        total_count = sum(class_counts)
        weighted_f1 = sum(
            (count / total_count) * f1 for f1, count in zip(f1_scores, class_counts)
        )
        return weighted_f1

    @classmethod
    def get_latency_metrics(
        cls,
        failing_test_results: List["DialogueUnderstandingTestResult"],
        passing_test_results: List["DialogueUnderstandingTestResult"],
    ) -> Dict[str, Dict[str, float]]:
        latencies = defaultdict(list)

        for result in failing_test_results + passing_test_results:
            for step in result.test_case.steps:
                if (
                    step.dialogue_understanding_output
                    and step.dialogue_understanding_output.latency
                ):
                    latencies["total"].append(
                        step.dialogue_understanding_output.latency
                    )
                for component_name, latency in step.get_latencies().items():
                    latencies[component_name].extend(latency)

        return {
            component_name: cls._calculate_percentiles(latency_list)
            for component_name, latency_list in latencies.items()
        }

    @classmethod
    def get_prompt_token_metrics(
        cls,
        failing_test_results: List["DialogueUnderstandingTestResult"],
        passing_test_results: List["DialogueUnderstandingTestResult"],
    ) -> Dict[str, Dict[str, float]]:
        tokens = defaultdict(list)

        for result in failing_test_results + passing_test_results:
            for step in result.test_case.steps:
                for component_name, token_count in step.get_prompt_tokens().items():
                    tokens[component_name].extend(token_count)

        return {
            component_name: cls._calculate_percentiles(latency_list)
            for component_name, latency_list in tokens.items()
        }

    @classmethod
    def get_completion_token_metrics(
        cls,
        failing_test_results: List["DialogueUnderstandingTestResult"],
        passing_test_results: List["DialogueUnderstandingTestResult"],
    ) -> Dict[str, Dict[str, float]]:
        tokens = defaultdict(list)

        for result in failing_test_results + passing_test_results:
            for step in result.test_case.steps:
                for component_name, token_count in step.get_completion_tokens().items():
                    tokens[component_name].extend(token_count)

        return {
            component_name: cls._calculate_percentiles(latency_list)
            for component_name, latency_list in tokens.items()
        }

    @staticmethod
    def _calculate_percentiles(values: List[float]) -> Dict[str, float]:
        return {
            "p50": float(np.percentile(values, 50)) if values else 0.0,
            "p90": float(np.percentile(values, 90)) if values else 0.0,
            "p99": float(np.percentile(values, 99)) if values else 0.0,
        }

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
