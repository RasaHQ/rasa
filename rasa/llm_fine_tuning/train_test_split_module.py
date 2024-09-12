import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Set, Tuple

import structlog

from rasa.e2e_test.e2e_test_case import TestSuite
from rasa.llm_fine_tuning.llm_data_preparation_module import LLMDataExample
from rasa.llm_fine_tuning.storage import StorageContext

TRAIN_TEST_MODULE_STORAGE_LOCATION = "4_train_test_split"

SUPPORTED_COMMANDS = [
    "SetSlot",
    "StartFlow",
    "CancelFlow",
    "ChitChat",
    "SkipQuestion",
    "SearchAndReply",
    "HumanHandoff",
    "Clarify",
]

INSTRUCTION_DATA_FORMAT = "instruction"
CONVERSATIONAL_DATA_FORMAT = "conversational"

KEY_COMMANDS = "commands"
KEY_DATA_EXAMPLES = "data_examples"
KEY_TEST_CASE_NAME = "test_case_name"

structlogger = structlog.get_logger()


# A protocol (similar to an interface) for data example that all fine-tuning data
# format classes should implement.
class DataExampleFormat(Protocol):
    def as_dict(self) -> dict: ...


@dataclass
class InstructionDataFormat(DataExampleFormat):
    # The prompt and completion fields are used to store the input and output of the
    # LLM model. The instruction data format is taken from the TRL library.
    # Refer: https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
    # The data format can also be used to train on Azure. Refer:
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/fine-tuning

    prompt: str
    completion: str

    def as_dict(self) -> Dict[str, str]:
        return {"prompt": self.prompt, "completion": self.completion}


@dataclass
class ConversationalMessageDataFormat:
    role: str
    content: str


@dataclass
class ConversationalDataFormat(DataExampleFormat):
    # The conversation data format is taken from the TRL library.
    # Refer: https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
    # The data format can also be used to train on Azure. Refer:
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/fine-tuning

    messages: List[ConversationalMessageDataFormat]

    def as_dict(self) -> Dict[str, List[Dict[str, str]]]:
        return {
            "messages": [
                {"role": message.role, "content": message.content}
                for message in self.messages
            ]
        }


def _get_command_types_covered_by_llm_data_point(commands: LLMDataExample) -> Set[str]:
    """Get the command types covered by the LLM data point.

    This function returns the set of command types from the output present in a
    LLMDataExample object. Eg: The function returns {'SetSlot', 'StartFlow'} when the
    LLMDataExample.output is 'SetSlot(slot, abc), SetSlot(slot, cde), StartFlow(xyz)'.
    """
    commands_covered = set()
    for command in SUPPORTED_COMMANDS:
        if command in commands.output:
            commands_covered.add(command)
    return commands_covered


def _get_grouped_fine_tuning_data_by_test_name_with_commands(
    fine_tuning_data: List[LLMDataExample],
) -> List[Dict[str, Any]]:
    """Group the fine-tune data by test name and capture commands covered by each group.

    This function groups the fine-tuning data by original_test_name and captures the
    commands covered by each group. The function returns a dictionary where the keys
    are the original_test_name and the values are dictionaries containing the data
    and commands covered by each group.
    """
    grouped_data: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {KEY_DATA_EXAMPLES: [], KEY_COMMANDS: set()}
    )

    for data_point in fine_tuning_data:
        grouped_data[data_point.original_test_name][KEY_DATA_EXAMPLES].append(
            data_point
        )
        command_types = _get_command_types_covered_by_llm_data_point(data_point)
        for command_type in command_types:
            grouped_data[data_point.original_test_name][KEY_COMMANDS].add(command_type)

    # Convert the grouped data to the required format.
    result = [
        {
            KEY_TEST_CASE_NAME: test_case_name,
            KEY_DATA_EXAMPLES: group[KEY_DATA_EXAMPLES],
            KEY_COMMANDS: group[KEY_COMMANDS],
        }
        for test_case_name, group in grouped_data.items()
    ]

    return result


def _get_minimum_test_case_groups_to_cover_all_commands(
    grouped_data: List[Dict[str, Any]],
) -> List[str]:
    """Get the minimum test case groups to cover all commands.

    This function implements a greedy algorithm that returns the minimum test case
    groups to cover all commands. The function takes a list of dictionaries where each
    dictionary represents a test group and contains the test_case_name, data_examples,
    and commands covered by each group.

    The function iteratively selects the test case group that covers the most number
    of uncovered commands until all commands are covered.

    The function returns the selected test case groups as a list of strings.

    Example:
    data = [
        {
            "test_case_name": "t1",
            "data_examples": [],
            "commands": {"SetSlot", "CancelFlow"}
        },
        {"test_case_name": "t2", "data_examples": [], "commands": {"CancelFlow"}},
        {"test_case_name": "t3", "data_examples": [], "commands": {"StartFlow"}},
        {
            "test_case_name": "t4",
            "data_examples": [],
            "commands": {"SetSlot", "StartFlow"}
        },
    ]

    selected_test_cases = _get_minimum_test_case_groups_to_cover_all_commands(data)
    print(selected_test_cases)
    # Output: ['t1', 't3']
    """
    # Get all the commands covered from the finetuning data.
    all_commands = set(
        command for test_group in grouped_data for command in test_group[KEY_COMMANDS]
    )
    selected_test_cases = []
    covered_commands: Set[str] = set()

    while covered_commands != all_commands:
        # Find the test case group that covers the most number of uncovered commands
        best_test_case_group = None
        commands_covered_by_best_test_case_group: Set[str] = set()

        for test_group in grouped_data:
            commands_covered = test_group[KEY_COMMANDS] - covered_commands
            if len(commands_covered) > len(commands_covered_by_best_test_case_group):
                best_test_case_group = test_group[KEY_TEST_CASE_NAME]
                commands_covered_by_best_test_case_group = commands_covered

        if best_test_case_group is None:
            break

        selected_test_cases.append(best_test_case_group)
        covered_commands.update(commands_covered_by_best_test_case_group)

    structlogger.info(
        "llm_fine_tuning.train_test_split_module.command_coverage_in_train_dataset",
        covered_commands=covered_commands,
    )
    return selected_test_cases


def _get_finetuning_data_in_instruction_data_format(
    train_data: List[Dict[str, Any]], validation_data: List[Dict[str, Any]]
) -> Tuple[List[DataExampleFormat], List[DataExampleFormat]]:
    """Convert the fine-tuning data to the required Alpaca format.

    The function takes the train and validation data as input and returns the formatted
    train and validation data.
    """

    def _convert_into_instruction_data_format(
        data: List[Dict[str, Any]],
    ) -> List[DataExampleFormat]:
        return [
            InstructionDataFormat(llm_data_example.prompt, llm_data_example.output)
            for test_group in data
            for llm_data_example in test_group[KEY_DATA_EXAMPLES]
        ]

    formatted_train_data = _convert_into_instruction_data_format(train_data)
    formatted_validation_data = _convert_into_instruction_data_format(validation_data)
    return formatted_train_data, formatted_validation_data


def _get_finetuning_data_in_conversational_data_format(
    train_data: List[Dict[str, Any]], validation_data: List[Dict[str, Any]]
) -> Tuple[List[DataExampleFormat], List[DataExampleFormat]]:
    """Convert the fine-tuning data to the required ShareGPT format.

    The function takes the train and validation data as input and returns the formatted
    train and validation data.
    """

    def _convert_into_conversational_data_format(
        data: List[Dict[str, Any]],
    ) -> List[DataExampleFormat]:
        return [
            ConversationalDataFormat(
                [
                    ConversationalMessageDataFormat("user", llm_data_example.prompt),
                    ConversationalMessageDataFormat(
                        "assistant", llm_data_example.output
                    ),
                ]
            )
            for test_group in data
            for llm_data_example in test_group[KEY_DATA_EXAMPLES]
        ]

    formatted_train_data = _convert_into_conversational_data_format(train_data)
    formatted_validation_data = _convert_into_conversational_data_format(
        validation_data
    )
    return formatted_train_data, formatted_validation_data


def _check_and_log_missing_validation_dataset_command_coverage(
    grouped_data: List[Dict[str, Any]],
    validation_data: List[Dict[str, Any]],
) -> None:
    """Check and log the missing commands in the validation data.

    Args:
        grouped_data: The grouped fine-tuning data - complete dataset.
        validation_data: The validation data to check for missing commands.
    """
    all_commands = set(
        command for test_group in grouped_data for command in test_group[KEY_COMMANDS]
    )
    command_coverage_in_validation_dataset = set(
        command
        for test_group in validation_data
        for command in test_group[KEY_COMMANDS]
    )

    missing_commands = all_commands - command_coverage_in_validation_dataset
    if missing_commands:
        structlogger.warning(
            "llm_fine_tuning.train_test_split_module.missing_commands_in_validation_dat"
            "aset",
            missing_commands=missing_commands,
        )


def _get_train_validation_data_splits(
    fine_tuning_data: List[LLMDataExample], train_frac: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Group by original_test_name and capture commands covered by each group.
    grouped_data = _get_grouped_fine_tuning_data_by_test_name_with_commands(
        fine_tuning_data
    )

    # Get the minimum test case groups to cover all commands. This is done to ensure
    # that the training data covers all the commands present in the fine-tuning data.
    selected_tests = _get_minimum_test_case_groups_to_cover_all_commands(grouped_data)

    # Separate the tests into train data and remaining data.
    train_data = [
        data for data in grouped_data if data[KEY_TEST_CASE_NAME] in selected_tests
    ]
    remaining_data = [
        data for data in grouped_data if data[KEY_TEST_CASE_NAME] not in selected_tests
    ]

    # Shuffle the remaining data randomly.
    random.shuffle(remaining_data)

    # Calculate the number of samples required for training
    total_samples = len(grouped_data)
    train_size = int(total_samples * train_frac)

    # Number of additional samples needed in train set
    current_train_size = len(train_data)
    additional_train_needed = train_size - current_train_size

    if additional_train_needed > 0:
        # Select additional samples from the remaining data
        additional_train_samples = remaining_data[:additional_train_needed]
        train_data.extend(additional_train_samples)

        # Update remaining data after adding to train data
        remaining_data = remaining_data[additional_train_needed:]

    # As the train dataset is selected based on the commands covered, the remaining data
    # might not have all the commands. So, we need check and throw a warning if any
    # command is missing in the remaining data. Ensuring command coverage over the
    # validation data will be handled in https://rasahq.atlassian.net/browse/ENG-1217.
    _check_and_log_missing_validation_dataset_command_coverage(
        grouped_data, remaining_data
    )
    return train_data, remaining_data


def split_e2e_test_suite(
    e2e_test_suite: TestSuite,
    train_data: List[Dict[str, Any]],
    validation_data: List[Dict[str, Any]],
) -> Tuple[TestSuite, TestSuite]:
    """Split the e2e test suite into train and validation test suites.

    Args:
        e2e_test_suite: The e2e test suite to split.
        train_data: The train data is used to get the names of the tests that should be
            included in the train test suite from the e2e test suite.
        validation_data: The validation data is used to get the names of the tests that
            should be included in the validation test suite from the e2e test suite.

    Returns:
        Tuple[TestSuite, TestSuite]: The train and validation test suites.
    """
    # Get the names of the test cases that should be included in the train and
    # validation test suites.
    train_test_case_names = {test[KEY_TEST_CASE_NAME] for test in train_data}
    validation_test_case_names = {test[KEY_TEST_CASE_NAME] for test in validation_data}

    train_test_cases = []
    validation_test_cases = []

    for test_case in e2e_test_suite.test_cases:
        if f"{test_case.file}::{test_case.name}" in train_test_case_names:
            train_test_cases.append(test_case)
        elif f"{test_case.file}::{test_case.name}" in validation_test_case_names:
            validation_test_cases.append(test_case)

    # Creating new TestSuite instances for train and validation
    train_suite = TestSuite(
        test_cases=train_test_cases,
        fixtures=e2e_test_suite.fixtures,
        metadata=e2e_test_suite.metadata,
        stub_custom_actions=e2e_test_suite.stub_custom_actions,
    )
    validation_suite = TestSuite(
        test_cases=validation_test_cases,
        fixtures=e2e_test_suite.fixtures,
        metadata=e2e_test_suite.metadata,
        stub_custom_actions=e2e_test_suite.stub_custom_actions,
    )

    return train_suite, validation_suite


def split_llm_fine_tuning_data(
    fine_tuning_data: List[LLMDataExample],
    train_frac: float,
    output_format: str,
    storage_context: StorageContext,
    e2e_test_suite: TestSuite,
) -> Tuple[List[DataExampleFormat], List[DataExampleFormat]]:
    # Split the fine-tuning data into train and validation data.
    train_data, validation_data = _get_train_validation_data_splits(
        fine_tuning_data, train_frac
    )
    if not validation_data:
        structlogger.warning(
            "llm_fine_tuning.train_test_split_module.empty_validation_dataset",
            event_info=(
                "Validation data is empty. Please provide more data to split into"
                " train and validation data. Also, check if the --train-frac value is"
                " appropriate."
            ),
            train_frac=train_frac,
        )

    # Convert the fine-tuning data to the required format.
    if output_format == INSTRUCTION_DATA_FORMAT:
        formatted_train_data, formatted_validation_data = (
            _get_finetuning_data_in_instruction_data_format(train_data, validation_data)
        )
    elif output_format == CONVERSATIONAL_DATA_FORMAT:
        formatted_train_data, formatted_validation_data = (
            _get_finetuning_data_in_conversational_data_format(
                train_data, validation_data
            )
        )
    else:
        raise ValueError(
            f"Output format '{output_format}' is not supported. Supported formats are "
            f"'{INSTRUCTION_DATA_FORMAT}' and '{CONVERSATIONAL_DATA_FORMAT}'."
        )

    # Write the train and validation data to the storage location.
    storage_context.write_formatted_finetuning_data(
        formatted_train_data,
        TRAIN_TEST_MODULE_STORAGE_LOCATION,
        "ft_splits/train.jsonl",
    )
    storage_context.write_formatted_finetuning_data(
        formatted_validation_data,
        TRAIN_TEST_MODULE_STORAGE_LOCATION,
        "ft_splits/val.jsonl",
    )

    # Split the e2e test suite data into train and validation data.
    train_suite, validation_suite = split_e2e_test_suite(
        e2e_test_suite, train_data, validation_data
    )

    # Write the train and validation test suite data to the storage location.
    storage_context.write_e2e_test_suite_to_yaml_file(
        train_suite, TRAIN_TEST_MODULE_STORAGE_LOCATION, "e2e_tests/train.yaml"
    )
    storage_context.write_e2e_test_suite_to_yaml_file(
        validation_suite,
        TRAIN_TEST_MODULE_STORAGE_LOCATION,
        "e2e_tests/validation.yaml",
    )

    return formatted_train_data, formatted_validation_data
