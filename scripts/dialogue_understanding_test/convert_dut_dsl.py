import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Any, Optional

import structlog
import yaml
from tqdm import tqdm

structlogger = structlog.get_logger()


DSL_MAPPINGS_KEY = "mappings"
FROM_DSL_REGEX_KEY = "from_dsl_regex"
TO_DSL_PATTERN_KEY = "to_dsl_pattern"
INPUT_SEPARATORS_KEY = "input_separators"
OUTPUT_SEPARATOR_KEY = "output_separator"

DUT_TEST_CASES_KEY = "test_cases"
DUT_TEST_CASE_KEY = "test_case"
DUT_TEST_CASE_STEPS_KEY = "steps"
DUT_TEST_CASE_COMMANDS_KEY = "commands"


@dataclass
class DSLMapping:
    """
    Defines a single DSL mapping rule.

    Attributes:
        from_dsl_regex: A regular expression used to identify and parse the old DSL
            command.

        to_dsl_pattern: A pattern used to construct the new DSL command.
            Supports placeholders like {1}, {2}, etc. for matched groups.

        input_separators: A list of input separators for matched group.
    """
    from_dsl_regex: str
    to_dsl_pattern: str
    input_separators: Optional[List[str]] = None
    output_separator: Optional[str] = None


def load_mapping_config(config_path: str) -> List[DSLMapping]:
    """
    Load the YAML file that follows this format:

    ```yaml

        mappings:
            - from_dsl_regex: <regular expression to match the old command>
              to_dsl_pattern: <string pattern containing placeholders {1}, {2}, ... >
              input_separators: <optional list of input separators>
              output_separator: <optional output separator>
            - from_dsl_regex: ...
              to_dsl_pattern: ...
            ...

    ```

    - **`from_dsl_regex`**: A regular expression (string) used to match the old DSL
      command. Must include any necessary anchors (e.g., `^` and `$`) and capturing
      groups `( ... )` for dynamic parts.

    - **`to_dsl_pattern`**: A string that contains placeholders like `{1}`,
      `{2}`, etc. Each placeholder corresponds to a capturing group in
      `from_dsl_regex`, in order of appearance. For example, if your regex  has two
      capturing groups, `{1}` inserts the content captured by the  first group, and
      `{2}` inserts the content captured by the second group.
    - **`input_separators`**: Optional list of input separators for matched group.
    - **`output_separator`**: Optional output separator to use for the options group.

    Returns:
        A list of parsed DSL mappings.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        yaml_mappings = yaml.safe_load(f)

    mappings = []
    for item in yaml_mappings.get(DSL_MAPPINGS_KEY, []):
        try:
            mappings.append(
                DSLMapping(
                    from_dsl_regex=item[FROM_DSL_REGEX_KEY],
                    to_dsl_pattern=item[TO_DSL_PATTERN_KEY],
                    input_separators=item.get(INPUT_SEPARATORS_KEY, None),
                    output_separator=item.get(OUTPUT_SEPARATOR_KEY, None),
                )
            )
        except Exception as e:
            structlogger.error(
                "convert_dut_dsl.load_dsl_mapping_config",
                event_info="Failed to load DSL mapping",
                error=e,
            )

    return mappings


def get_yaml_paths(test_dir: str) -> List[str]:
    """
    Finds all .yaml or .yml files in root_dir and returns
    a list of their absolute paths.

    Args:
        test_dir: The directory to search for test case files.
    Returns:
        A list of absolute file paths.
    """
    test_case_files = []
    for current_path, directories, files in os.walk(test_dir):
        for filename in files:
            if filename.lower().endswith((".yaml", ".yml")):
                absolute_path = os.path.abspath(
                    os.path.join(current_path, filename))
                test_case_files.append(absolute_path)
    return test_case_files


def replace_separators(
        s: str,
        input_separators: Optional[List[str]] = None,
        output_separator: Optional[str] = None,
) -> str:
    if not input_separators or not output_separator:
        return s

    # Regex pattern to match any separator
    pattern = "|".join(map(re.escape, input_separators))
    return re.sub(rf"(?:{pattern})+", output_separator, s).strip(output_separator)


def transform_command(command: str, mappings: List[DSLMapping]) -> str:
    """
    Attempts to transform the given command string into a new command string.

    Args:
        command: The command string to transform.
        mappings: A list of DSL mappings.

    Returns:
        - If matched, returns the new DSL string.
        - If no mapping matches, returns the command unchanged.
    """
    for mapping in mappings:
        match = re.match(mapping.from_dsl_regex, command)
        if match:
            # Match is found, build the new command string
            transformed = mapping.to_dsl_pattern
            for i, group in enumerate(match.groups(), start=1):
                # Replace {i} in new_pattern with group
                cleaned_group = replace_separators(
                    group,
                    mapping.input_separators,
                    mapping.output_separator
                )
                transformed = transformed.replace(f"{{{i}}}", cleaned_group)
            return transformed

    # No mapping is matched, return the command as-is
    return command


def transform_test_case(test_case: dict, mappings: List[DSLMapping]) -> dict:
    """
    Transform the commands in a single test case.
    """
    steps = test_case.get(DUT_TEST_CASE_STEPS_KEY, [])

    for step in steps:

        # If the step doesn't contain commands key, skip it (utter step)
        if DUT_TEST_CASE_COMMANDS_KEY not in step:
            continue

        # Attempt to parse the commands
        new_commands = []
        for command in step[DUT_TEST_CASE_COMMANDS_KEY]:
            new_command = transform_command(command, mappings)
            new_commands.append(new_command)

        step[DUT_TEST_CASE_COMMANDS_KEY] = new_commands

    return test_case


def transform_yaml_data(
    input_path: str,
    mappings: List
) -> Any:
    """
    Reads YAML data from input_path.
    If the file contains a 'test_cases' key, transform each test case
    using the provided mappings.
    Returns the transformed data (a dict) or None if empty/unparseable.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # If there's nothing to transform, return data as-is
    if not data or DUT_TEST_CASES_KEY not in data:
        return data

    # Transform each test case
    data[DUT_TEST_CASES_KEY] = [
        transform_test_case(test_case, mappings)
        for test_case in data[DUT_TEST_CASES_KEY]
    ]
    return data


def write_transformed_data(
    data: Any,
    output_path: str,
) -> None:
    """
    Writes the given data (dict) out to the specified output_path (in YAML).
    Ensures necessary directories are created.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert commands in YAML with Dialogue Understanding Tests"
            "(DUTs) to a new DSL."
        )
    )
    parser.add_argument(
        "--dut-tests-dir",
        required=True,
        help="Path to the input directory containing DUT tests."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to the output directory for transformed DUT tests."
    )
    parser.add_argument(
        "--dsl-mappings",
        required=True,
        help=(
            "Path to the YAML file with DSL mappings.\n\n"
            "The YAML file should have a structure like:\n\n"
            "```yaml\n"
            "mappings:\n\n"
            "  - from_dsl_regex: <regular expression to match the old command>\n"
            "    to_dsl_pattern: <string pattern containing placeholders {1}, {2}, ... >\n\n"  # noqa: E501
            "    [optional] input_separators: <list of separator strings>\n"
            "    [optional] output_separator: <output_separator>\n"
            "  - from_dsl_regex: ...\n"
            "    to_dsl_pattern: ...\n"
            "  ...\n"
            "```\n\n"
            "- **`from_dsl_regex`**: A regular expression (string) used to match the old DSL\n"  # noqa: E501
            "  command. Must include any necessary anchors (e.g., `^` and `$`) and capturing\n"  # noqa: E501
            "  groups `( ... )` for dynamic parts.\n\n"
            "- **`to_dsl_pattern`**: A string that contains placeholders like `{1}`, `{2}`, etc.\n"  # noqa: E501
            "  Each placeholder corresponds to a capturing group in `from_dsl_regex`, in order\n"  # noqa: E501
            "  of appearance. For example, if your regex has two capturing groups, `{1}` inserts\n"  # noqa: E501
            "  the content captured by the first group, and `{2}` inserts the content captured\n"  # noqa: E501
            "  by the second group.\n"
            "- **`input_separators`**: A list of input separators for matched group.\n"
            "- **`output_separator`**: The output separator to use for the options group.\n"
        )
    )

    args = parser.parse_args()

    dut_tests_dir = os.path.abspath(args.dut_tests_dir)
    output_dir = os.path.abspath(args.output_dir)
    dsl_mappings = os.path.abspath(args.dsl_mappings)

    # Load mappings
    mappings = load_mapping_config(dsl_mappings)

    # Get the absolute paths of the .YAML files with DUT tests from the given directory
    yaml_file_paths = get_yaml_paths(dut_tests_dir)

    for yaml_file_path in tqdm(
        yaml_file_paths,
        desc="Processing YAML files",
        unit="file"
    ):
        # Get the portion of the file path relative to dut_tests_dir
        relative_yaml_file_path = os.path.relpath(yaml_file_path, start=dut_tests_dir)
        # Build the corresponding path under output_dir
        output_path = os.path.join(output_dir, relative_yaml_file_path)
        # Transform the YAML
        data = transform_yaml_data(yaml_file_path, mappings)

        if data is None or DUT_TEST_CASES_KEY not in data:
            structlogger.info(
                f"{yaml_file_path}: Nothing to transform."
                f"Output written to: {output_path}",
                input_path=yaml_file_path,
                output_path=output_path,
            )
        else:
            structlogger.info(
                f"{yaml_file_path}: Transformation complete."
                f"Output written to: {output_path}",
                input_path=yaml_file_path,
                output_path=output_path,
            )

        # Write the data
        write_transformed_data(data, output_path)

    structlogger.info(f"Transformation complete. Output written to: {output_dir}")


if __name__ == '__main__':
    main()
