from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import ruamel

from rasa.e2e_test.e2e_yaml_utils import E2EYAMLWriter


def test_e2e_write_tests_to_yaml_creates_file(tmp_path: Path):
    writer = E2EYAMLWriter(output_path=str(tmp_path))
    writer.write_to_file("")

    output_files = list(tmp_path.glob("e2e_tests_*.yml"))
    assert len(output_files) == 1


def test_e2e_write_tests_to_yaml_correct_content(tmp_path: Path):
    writer = E2EYAMLWriter(output_path=str(tmp_path))
    test_yaml_string = """
    - test_case: user_greeting_the_assistant
      steps:
        - user: "Hi"
          assertions:
            - bot_uttered:
                text_matches: "Hello"
    """
    writer.write_to_file(test_yaml_string)

    output_files = list(tmp_path.glob("e2e_tests_*.yml"))
    with open(output_files[0], "r") as file:
        written_content = file.read()

    written_yaml_data = ruamel.yaml.safe_load(written_content)
    test_yaml_data = ruamel.yaml.safe_load(test_yaml_string)

    assert written_yaml_data == [{"test_cases": test_yaml_data}]


def test_e2e_write_tests_to_yaml_creates_directory(tmp_path: Path):
    output_path = tmp_path / "nested"
    writer = E2EYAMLWriter(output_path=str(output_path))
    test_yaml_string = """
    - test_case: user_greeting_the_assistant
      steps:
        - user: "Hi"
          assertions:
            - bot_uttered:
                text_matches: "Hello"
    """
    writer.write_to_file(test_yaml_string)
    assert output_path.exists()


def test_e2e_write_tests_to_yaml_correct_timestamp(tmp_path: Path):
    writer = E2EYAMLWriter(output_path=str(tmp_path))

    fixed_timestamp = datetime(2024, 1, 1, 12, 0, 0)
    expected_timestamp_str = fixed_timestamp.strftime("%Y%m%d_%H%M%S")

    with patch("rasa.e2e_test.e2e_yaml_utils.datetime") as mock_datetime:
        mock_datetime.now.return_value = fixed_timestamp
        mock_datetime.now.strftime = datetime.strftime
        writer.write_to_file("")
        output_files = list(tmp_path.glob(f"e2e_tests_{expected_timestamp_str}.yml"))
        assert len(output_files) == 1
