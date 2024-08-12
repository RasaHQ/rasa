from typing import List
from unittest.mock import call, MagicMock

import pytest
import structlog

from rasa.cli.e2e_test import read_test_cases
from rasa.e2e_test.e2e_test_case import TestSuite
from rasa.llm_fine_tuning.llm_data_preparation_module import LLMDataExample
from rasa.llm_fine_tuning.train_test_split_module import (
    _get_minimum_test_case_groups_to_cover_all_commands,
    split_llm_fine_tuning_data,
    InstructionDataFormat,
    ConversationalDataFormat,
    ConversationalMessageDataFormat,
    INSTRUCTION_DATA_FORMAT,
    CONVERSATIONAL_DATA_FORMAT,
)
from tests.utilities import filter_logs


@pytest.fixture()
def test_suite() -> TestSuite:
    return read_test_cases("data/test_llm_finetuning/e2e_test_sample.yml")


@pytest.fixture()
def llm_fine_tuning_data() -> List[LLMDataExample]:
    return [
        LLMDataExample(
            original_test_name="e2e_sample_test.yml::test1",
            original_user_utterance="user_utterance_1",
            output="StartFlow(abc), SetSlot(xyz, temp)",
            prompt="test_prompt has rephrased_user_utterance_1",
            rephrased_user_utterance="rephrased_user_utterance_1",
        ),
        LLMDataExample(
            original_test_name="e2e_sample_test.yml::test1",
            original_user_utterance="user_utterance_1",
            output="StartFlow(abc), SetSlot(xyz, temp)",
            prompt="test_prompt has rephrased_user_utterance_2",
            rephrased_user_utterance="rephrased_user_utterance_2",
        ),
        LLMDataExample(
            original_test_name="e2e_sample_test.yml::test2",
            original_user_utterance="user_utterance_2",
            output="StartFlow(def)",
            prompt="test_prompt has rephrased_user_utterance_A",
            rephrased_user_utterance="rephrased_user_utterance_A",
        ),
        LLMDataExample(
            original_test_name="e2e_sample_test.yml::test2",
            original_user_utterance="user_utterance_2",
            output="StartFlow(def)",
            prompt="test_prompt has rephrased_user_utterance_B",
            rephrased_user_utterance="rephrased_user_utterance_B",
        ),
    ]


def test_split_llm_fine_tuning_data_train_alpaca_format_50_percent_split_frac(
    llm_fine_tuning_data: List[LLMDataExample], test_suite: TestSuite
) -> None:
    # Given
    storage_context = MagicMock()
    with structlog.testing.capture_logs() as caplog:
        train_data, val_data = split_llm_fine_tuning_data(
            llm_fine_tuning_data,
            0.5,
            INSTRUCTION_DATA_FORMAT,
            storage_context,
            test_suite,
        )

    assert len(train_data) == 2
    assert len(val_data) == 2
    assert train_data == (
        [
            InstructionDataFormat(
                prompt="test_prompt has rephrased_user_utterance_1",
                completion="StartFlow(abc), SetSlot(xyz, temp)",
            ),
            InstructionDataFormat(
                prompt="test_prompt has rephrased_user_utterance_2",
                completion="StartFlow(abc), SetSlot(xyz, temp)",
            ),
        ]
    )
    assert val_data == (
        [
            InstructionDataFormat(
                prompt="test_prompt has rephrased_user_utterance_A",
                completion="StartFlow(def)",
            ),
            InstructionDataFormat(
                prompt="test_prompt has rephrased_user_utterance_B",
                completion="StartFlow(def)",
            ),
        ]
    )

    # Filter logs for missing commands in validation dataset.
    logs = filter_logs(
        caplog,
        event=(
            "llm_fine_tuning.train_test_split_module.missing_commands_in_validation_dat"
            "aset"
        ),
        log_level="warning",
    )
    assert len(logs) == 1
    assert logs[0]["missing_commands"] == {"SetSlot"}

    # Check for the expected calls to write the formatted fine-tuning data to storage.
    expected_calls = [
        call(train_data, "4_train_test_split", "ft_splits/train.jsonl"),
        call(val_data, "4_train_test_split", "ft_splits/val.jsonl"),
    ]
    storage_context.write_formatted_finetuning_data.assert_has_calls(
        expected_calls, any_order=False
    )


def test_split_llm_fine_tuning_data_train_alpaca_format_100_percent_split_frac(
    llm_fine_tuning_data: List[LLMDataExample], test_suite: TestSuite
) -> None:
    # Given
    storage_context = MagicMock()

    with structlog.testing.capture_logs() as caplog:
        train_data, val_data = split_llm_fine_tuning_data(
            llm_fine_tuning_data,
            1.0,
            INSTRUCTION_DATA_FORMAT,
            storage_context,
            test_suite,
        )

    assert len(train_data) == 4
    assert len(val_data) == 0
    assert train_data == (
        [
            InstructionDataFormat(
                prompt="test_prompt has rephrased_user_utterance_1",
                completion="StartFlow(abc), SetSlot(xyz, temp)",
            ),
            InstructionDataFormat(
                prompt="test_prompt has rephrased_user_utterance_2",
                completion="StartFlow(abc), SetSlot(xyz, temp)",
            ),
            InstructionDataFormat(
                prompt="test_prompt has rephrased_user_utterance_A",
                completion="StartFlow(def)",
            ),
            InstructionDataFormat(
                prompt="test_prompt has rephrased_user_utterance_B",
                completion="StartFlow(def)",
            ),
        ]
    )
    # Filter logs for missing commands in validation dataset.
    logs = filter_logs(
        caplog,
        event=(
            "llm_fine_tuning.train_test_split_module.missing_commands_in_validation_data"
            "set"
        ),
        log_level="warning",
    )
    assert len(logs) == 1
    assert logs[0]["missing_commands"] == {"SetSlot", "StartFlow"}

    # Filter logs for empty validation dataset.
    logs = filter_logs(
        caplog,
        event="llm_fine_tuning.train_test_split_module.empty_validation_dataset",
        log_level="warning",
    )
    assert len(logs) == 1
    assert logs[0]["train_frac"] == 1.0

    # Check for the expected calls to write the formatted fine-tuning data to storage.
    expected_calls = [
        call(train_data, "4_train_test_split", "ft_splits/train.jsonl"),
        call(val_data, "4_train_test_split", "ft_splits/val.jsonl"),
    ]
    storage_context.write_formatted_finetuning_data.assert_has_calls(
        expected_calls, any_order=False
    )


def test_split_llm_fine_tuning_data_train_sharegpt_format(
    llm_fine_tuning_data: List[LLMDataExample], test_suite: TestSuite
) -> None:
    # Given
    storage_context = MagicMock()
    with structlog.testing.capture_logs() as caplog:
        train_data, val_data = split_llm_fine_tuning_data(
            llm_fine_tuning_data,
            0.5,
            CONVERSATIONAL_DATA_FORMAT,
            storage_context,
            test_suite,
        )

    assert len(train_data) == 2
    assert len(val_data) == 2
    assert train_data == (
        [
            ConversationalDataFormat(
                messages=[
                    ConversationalMessageDataFormat(
                        role="user",
                        content="test_prompt has rephrased_user_utterance_1",
                    ),
                    ConversationalMessageDataFormat(
                        role="assistant", content="StartFlow(abc), SetSlot(xyz, temp)"
                    ),
                ]
            ),
            ConversationalDataFormat(
                messages=[
                    ConversationalMessageDataFormat(
                        role="user",
                        content="test_prompt has rephrased_user_utterance_2",
                    ),
                    ConversationalMessageDataFormat(
                        role="assistant", content="StartFlow(abc), SetSlot(xyz, temp)"
                    ),
                ]
            ),
        ]
    )
    assert val_data == (
        [
            ConversationalDataFormat(
                messages=[
                    ConversationalMessageDataFormat(
                        role="user",
                        content="test_prompt has rephrased_user_utterance_A",
                    ),
                    ConversationalMessageDataFormat(
                        role="assistant", content="StartFlow(def)"
                    ),
                ]
            ),
            ConversationalDataFormat(
                messages=[
                    ConversationalMessageDataFormat(
                        role="user",
                        content="test_prompt has rephrased_user_utterance_B",
                    ),
                    ConversationalMessageDataFormat(
                        role="assistant", content="StartFlow(def)"
                    ),
                ]
            ),
        ]
    )
    # Filter logs for missing commands in validation dataset.
    logs = filter_logs(
        caplog,
        event=(
            "llm_fine_tuning.train_test_split_module.missing_commands_in_validation_dat"
            "aset"
        ),
        log_level="warning",
    )
    assert len(logs) == 1
    assert logs[0]["missing_commands"] == {"SetSlot"}

    # Check for the expected calls to write the formatted fine-tuning data to storage.
    expected_calls = [
        call(train_data, "4_train_test_split", "ft_splits/train.jsonl"),
        call(val_data, "4_train_test_split", "ft_splits/val.jsonl"),
    ]
    storage_context.write_formatted_finetuning_data.assert_has_calls(
        expected_calls, any_order=False
    )


def test_split_llm_fine_tuning_data_train_unsupported_format(
    llm_fine_tuning_data: List[LLMDataExample], test_suite: TestSuite
) -> None:
    # Given
    storage_context = MagicMock()
    with pytest.raises(ValueError) as e:
        split_llm_fine_tuning_data(
            llm_fine_tuning_data,
            0.5,
            "some_format",
            storage_context,
            test_suite,
        )

    assert str(e.value) == (
        f"Output format 'some_format' is not supported. Supported formats are "
        f"'{INSTRUCTION_DATA_FORMAT}' and '{CONVERSATIONAL_DATA_FORMAT}'."
    )


def test_get_minimum_test_case_groups_to_cover_all_commands_selects_few_cases() -> None:
    # Given
    grouped_data = [
        {
            "test_case_name": "t1",
            "data_examples": [],
            "commands": {"StartFlow", "CancelFlow", "HumanHandoff"},
        },
        {"test_case_name": "t2", "data_examples": [], "commands": {"CancelFlow"}},
        {"test_case_name": "t3", "data_examples": [], "commands": {"SetSlot"}},
        {
            "test_case_name": "t4",
            "data_examples": [],
            "commands": {"SetSlot", "StartFlow"},
        },
        {
            "test_case_name": "t5",
            "data_examples": [],
            "commands": {"SetSlot", "HumanHandoff"},
        },
    ]

    # When
    result = _get_minimum_test_case_groups_to_cover_all_commands(grouped_data)

    # Then
    assert result == ["t1", "t3"]


def test_get_minimum_test_case_groups_to_cover_all_commands_selects_all_cases() -> None:
    # Given
    grouped_data = [
        {"test_case_name": "t1", "data_examples": [], "commands": {"StartFlow"}},
        {"test_case_name": "t2", "data_examples": [], "commands": {"CancelFlow"}},
        {"test_case_name": "t3", "data_examples": [], "commands": {"SetSlot"}},
        {"test_case_name": "t4", "data_examples": [], "commands": {"Clarify"}},
        {"test_case_name": "t5", "data_examples": [], "commands": {"HumanHandoff"}},
    ]

    # When
    result = _get_minimum_test_case_groups_to_cover_all_commands(grouped_data)

    # Then
    assert result == ["t1", "t2", "t3", "t4", "t5"]


def test_get_minimum_test_case_groups_to_cover_all_commands_selects_first_case() -> (
    None
):
    # Given
    grouped_data = [
        {
            "test_case_name": "t1",
            "data_examples": [],
            "commands": {"StartFlow", "CancelFlow", "HumanHandoff", "SetSlot"},
        },
        {"test_case_name": "t2", "data_examples": [], "commands": {"CancelFlow"}},
        {"test_case_name": "t3", "data_examples": [], "commands": {"SetSlot"}},
        {
            "test_case_name": "t4",
            "data_examples": [],
            "commands": {"SetSlot", "StartFlow"},
        },
        {
            "test_case_name": "t5",
            "data_examples": [],
            "commands": {"SetSlot", "HumanHandoff"},
        },
    ]

    # When
    result = _get_minimum_test_case_groups_to_cover_all_commands(grouped_data)

    # Then
    assert result == ["t1"]


def test_get_minimum_test_case_groups_to_cover_all_commands_selects_last_case() -> None:
    # Given
    grouped_data = [
        {"test_case_name": "t1", "data_examples": [], "commands": {"CancelFlow"}},
        {"test_case_name": "t2", "data_examples": [], "commands": {"SetSlot"}},
        {
            "test_case_name": "t3",
            "data_examples": [],
            "commands": {"SetSlot", "StartFlow"},
        },
        {
            "test_case_name": "t4",
            "data_examples": [],
            "commands": {"SetSlot", "HumanHandoff"},
        },
        {
            "test_case_name": "t5",
            "data_examples": [],
            "commands": {"StartFlow", "CancelFlow", "HumanHandoff", "SetSlot"},
        },
    ]

    # When
    result = _get_minimum_test_case_groups_to_cover_all_commands(grouped_data)

    # Then
    assert result == ["t5"]
