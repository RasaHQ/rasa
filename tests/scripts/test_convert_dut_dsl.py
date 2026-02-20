import os
from typing import List
from unittest.mock import Mock, patch

import pytest
import yaml

from rasa.dialogue_understanding.commands import (
    CancelFlowCommand,
    ChitChatAnswerCommand,
    ClarifyCommand,
    KnowledgeAnswerCommand,
    SetSlotCommand,
    StartFlowCommand,
)
from scripts.dialogue_understanding_test.convert_dut_dsl import (
    DSLMapping,
    get_yaml_paths,
    load_mapping_config,
    main,
    transform_command,
)


@pytest.fixture
def mappings() -> dict:
    data = {
        "mappings": [
            # StartFlow(flow_name) -> start flow_name
            {
                "from_dsl_regex": StartFlowCommand.regex_pattern(),
                "to_dsl_pattern": "start {1}",
            },
            # SetSlot(slot_name slot_value) -> set slot_name slot_value
            {
                "from_dsl_regex": SetSlotCommand.regex_pattern(),
                "to_dsl_pattern": "set {1} {2}",
            },
            # CancelFlow() -> cancel
            {
                "from_dsl_regex": CancelFlowCommand.regex_pattern(),
                "to_dsl_pattern": "cancel",
            },
            # Clarify(...) -> clarify ...
            {
                "from_dsl_regex": ClarifyCommand.regex_pattern(),
                "to_dsl_pattern": "clarify {1}",
            },
            # SearchAndReply() -> search
            {
                "from_dsl_regex": KnowledgeAnswerCommand.regex_pattern(),
                "to_dsl_pattern": "search",
            },
            # ChitChat -> chat
            {
                "from_dsl_regex": ChitChatAnswerCommand.regex_pattern(),
                "to_dsl_pattern": "chat",
            },
        ]
    }
    return data


@pytest.fixture
def mappings_parsed(mappings) -> List[DSLMapping]:
    parsed = []
    for mapping in mappings["mappings"]:
        parsed.append(DSLMapping(**mapping))
    return parsed


def test_load_valid_mapping(mappings: dict, tmp_path):
    """Test that a valid mapping file is parsed correctly using pytest's tmp_path."""
    with open(tmp_path / "mappings.yaml", "w", encoding="utf-8") as f:
        yaml.dump(mappings, f, sort_keys=False)

    loaded_mappings = load_mapping_config(str(tmp_path / "mappings.yaml"))

    assert isinstance(loaded_mappings, list)
    assert len(loaded_mappings) == 6
    assert isinstance(loaded_mappings[0], DSLMapping)
    assert isinstance(loaded_mappings[1], DSLMapping)

    assert loaded_mappings[0].from_dsl_regex == StartFlowCommand.regex_pattern()
    assert loaded_mappings[0].to_dsl_pattern == "start {1}"
    assert loaded_mappings[1].from_dsl_regex == SetSlotCommand.regex_pattern()
    assert loaded_mappings[1].to_dsl_pattern == "set {1} {2}"


def test_get_yaml_paths(tmp_path):
    # Create test directory structure
    root_dit = tmp_path / "root"
    root_dit.mkdir()
    sub_dir = root_dit / "sub_dir"
    sub_dir.mkdir()

    # Create YAML files
    yaml_file_1 = root_dit / "test_1.yaml"
    yaml_file_2 = root_dit / "test_2.yml"
    yaml_file_3 = sub_dir / "nested_test.yaml"

    yaml_file_1.write_text("dummy content", encoding="utf-8")
    yaml_file_2.write_text("dummy content", encoding="utf-8")
    yaml_file_3.write_text("dummy content", encoding="utf-8")

    # Create a non-YAML file
    non_yaml_file = tmp_path / "ignore.txt"
    non_yaml_file.write_text("This should be ignored", encoding="utf-8")

    # Run the function
    found_files = get_yaml_paths(str(tmp_path))

    # Expected absolute file paths
    expected_files = {str(yaml_file_1), str(yaml_file_2), str(yaml_file_3)}

    # Check that only the YAML files were found
    assert set(found_files) == expected_files


@pytest.mark.parametrize(
    "input_command, transformed_command",
    [
        ("StartFlow(flow_name)", "start flow_name"),
        ("SetSlot(slot_name, slot_value)", "set slot_name slot_value"),
        ("Clarify(flow_a, flow_b, flow_c)", "clarify flow_a, flow_b, flow_c"),
    ],
)
def test_transform_command(
    mappings_parsed: List[DSLMapping], input_command: str, transformed_command: str
):
    actual_transformed = transform_command(input_command, mappings_parsed)
    assert actual_transformed == transformed_command


@pytest.mark.parametrize(
    "input_command, transformed_command",
    [
        ("Clarify(flow_a, flow_b, flow_c)", "clarify flow_a flow_b flow_c"),
        ("Clarify(flow_a flow_b flow_c)", "clarify flow_a flow_b flow_c"),
    ],
)
def test_replace_separators(input_command: str, transformed_command: str):
    # Add mapping
    data = {
        "mappings": [
            # Clarify(...) -> clarify ...
            {
                "from_dsl_regex": ClarifyCommand.regex_pattern(),
                "to_dsl_pattern": "clarify {1}",
                "input_separators": [" ", ","],
                "output_separator": " ",
            },
        ]
    }
    mappings_parsed = []
    for mapping in data["mappings"]:
        mappings_parsed.append(DSLMapping(**mapping))

    actual_transformed = transform_command(input_command, mappings_parsed)
    assert actual_transformed == transformed_command


@patch("scripts.dialogue_understanding_test.convert_dut_dsl.load_mapping_config")
@patch("scripts.dialogue_understanding_test.convert_dut_dsl.get_yaml_paths")
@patch(
    "scripts.dialogue_understanding_test.convert_dut_dsl.transform_yaml_data",
    return_value={"test_cases": []},
)
@patch("argparse.ArgumentParser.parse_args")
def test_main_preserves_directory_structure(
    mock_args,
    mock_transform_yaml_data,
    mock_get_yaml_paths,
    mock_load_mapping_config,
    tmp_path,
    mappings_parsed,
):
    # Setup test directories
    input_dir = tmp_path / "dut_tests"
    output_dir = tmp_path / "output_tests"

    subdir_1 = input_dir / "subfolder_1"
    subdir_2 = input_dir / "subfolder_2"
    subdir_1.mkdir(parents=True)
    subdir_2.mkdir(parents=True)

    yaml_file_1 = subdir_1 / "test_1.yaml"
    yaml_file_2 = subdir_2 / "test_2.yml"
    yaml_file_3 = input_dir / "root_test.yaml"

    yaml_file_1.write_text("dummy content", encoding="utf-8")
    yaml_file_2.write_text("dummy content", encoding="utf-8")
    yaml_file_3.write_text("dummy content", encoding="utf-8")

    yaml_files = [yaml_file_1, yaml_file_2, yaml_file_3]

    # Mock CLI arguments
    mock_args.return_value = Mock(
        dut_tests_dir=input_dir,
        output_dir=output_dir,
        dsl_mappings="mock_mappings.yaml",
    )

    # Mock get_yaml_paths() to return our test YAML files
    mock_get_yaml_paths.return_value = yaml_files

    # Mock load mappings config
    mock_load_mapping_config.return_value = mappings_parsed

    # Run main()
    main()

    # Check that all expected directories are recreated in the output folder
    for yaml_file in yaml_files:
        relative_path = os.path.relpath(yaml_file, start=input_dir)
        expected_output_path = os.path.join(output_dir, relative_path)
        expected_output_dir = os.path.dirname(expected_output_path)

        assert os.path.exists(expected_output_dir)

    # Ensure transform and write were called the expected number of times
    assert mock_transform_yaml_data.call_count == len(yaml_files)
