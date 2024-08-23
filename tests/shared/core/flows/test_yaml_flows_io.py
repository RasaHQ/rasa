import os
import tempfile
import textwrap

import pytest

from rasa.shared.core.flows.yaml_flows_io import (
    is_flows_file,
    YAMLFlowsReader,
    YamlFlowsWriter,
)
from rasa.shared.utils.yaml import YamlValidationException


@pytest.fixture(scope="module")
def basic_flows_file(tests_data_folder: str) -> str:
    return os.path.join(tests_data_folder, "test_flows", "basic_flows.yml")


@pytest.fixture(scope="module")
def flows_with_metadata_file(tests_data_folder: str) -> str:
    return os.path.join(tests_data_folder, "test_flows", "flows_with_metadata.yml")


@pytest.mark.parametrize(
    "path, expected_result",
    [
        (os.path.join("test_flows", "basic_flows.yml"), True),
        (os.path.join("test_moodbot", "domain.yml"), False),
    ],
)
def test_is_flows_file(tests_data_folder: str, path: str, expected_result: bool):
    full_path = os.path.join(tests_data_folder, path)
    assert is_flows_file(full_path) == expected_result


def test_flow_reading(basic_flows_file: str):
    flows_list = YAMLFlowsReader.read_from_file(basic_flows_file)
    assert len(flows_list) == 2
    assert flows_list.flow_by_id("foo") is not None
    assert flows_list.flow_by_id("bar") is not None


def test_flow_writing(basic_flows_file: str):
    flows_list = YAMLFlowsReader.read_from_file(basic_flows_file)
    _, tmp_file_name = tempfile.mkstemp()
    YamlFlowsWriter.dump(flows_list.underlying_flows, tmp_file_name)

    re_read_flows_list = YAMLFlowsReader.read_from_file(
        tmp_file_name, add_line_numbers=False
    )
    assert re_read_flows_list.as_json_list() == flows_list.as_json_list()


def test_flow_writing_double_metadata(flows_with_metadata_file: str):
    flows_list = YAMLFlowsReader.read_from_file(flows_with_metadata_file)
    _, tmp_file_name = tempfile.mkstemp()
    YamlFlowsWriter.dump(flows_list.underlying_flows, tmp_file_name)

    re_read_flows_list = YAMLFlowsReader.read_from_file(tmp_file_name)
    assert re_read_flows_list.as_json_list() == flows_list.as_json_list()


def test_flow_validate_invalid_else():
    data = textwrap.dedent(
        """
        flows:
            add_contact:
                description: add a contact to your contact list
                name: add a contact
                steps:
                - collect: "add_contact_handle"
                  next:
                  - if: "slots.return_value = 'success'"
                    then: END
                  - else:
                        action: utter_add_contact_error
                        next: END
        """
    )
    with pytest.raises(YamlValidationException) as e:
        YAMLFlowsReader.read_from_string(data)
    assert "Not a valid 'else' definition. Expected list of steps or step id." in str(
        e.value
    )


def test_flow_validate_ambiguous_step_type():
    data = textwrap.dedent(
        """
        flows:
            add_contact:
                description: add a contact to your contact list
                steps:
                - collect: "add_contact_handle"
                  action: add_contact
        """
    )
    with pytest.raises(YamlValidationException) as e:
        YAMLFlowsReader.read_from_string(data)
    expected_error = "Additional properties are not allowed ('collect' was unexpected)"
    assert expected_error in str(e.value)


def test_flow_validate_wrong_type():
    data = textwrap.dedent(
        """
        flows:
            add_contact:
                description: add a contact to your contact list
                steps:
                - collect: "add_contact_handle"
                  ask_before_filling: 42
        """
    )
    with pytest.raises(YamlValidationException) as e:
        YAMLFlowsReader.read_from_string(data)
    assert "Found `42` but expected a boolean." in str(e.value)


def test_flow_validate_invalid_next():
    data = textwrap.dedent(
        """
        flows:
            add_contact:
                description: add a contact to your contact list
                steps:
                - collect: "add_contact_handle"
                  next: 42
        """
    )
    with pytest.raises(YamlValidationException) as e:
        YAMLFlowsReader.read_from_string(data)
    assert (
        "Not a valid 'next' definition. Expected list of conditions or step id."
        in str(e.value)
    )


def test_flow_validate_invalid_set_slots():
    # set slots should be an array rather than a dict
    data = textwrap.dedent(
        """
        flows:
            add_contact:
                description: add a contact to your contact list
                steps:
                - set_slots:
                    foo: bar
        """
    )
    with pytest.raises(YamlValidationException) as e:
        YAMLFlowsReader.read_from_string(data)
    assert "Found a dictionary but expected a list of slot sets." in str(e.value)


def test_flow_validate_invalid_next_list():
    data = textwrap.dedent(
        """
        flows:
            add_contact:
                description: add a contact to your contact list
                steps:
                - collect: "foo"
                  next:
                  - collect: "bar"
                    next: 42
        """
    )
    with pytest.raises(YamlValidationException) as e:
        YAMLFlowsReader.read_from_string(data)
    assert (
        "Not a valid 'next' definition. Expected else block or if-then block."
        in str(e.value)
    )


def test_flow_validate_invalid_nested_next():
    data = textwrap.dedent(
        """
        flows:
            add_contact:
                description: add a contact to your contact list
                steps:
                - collect: "foo"
                  next:
                  - if: true
                    then:
                    - collect: "bar"
                      next: 42   # invalid
                  - else: END
        """
    )
    with pytest.raises(YamlValidationException) as e:
        YAMLFlowsReader.read_from_string(data)
    assert (
        "Not a valid 'next' definition. Expected list of conditions or step id."
        in str(e.value)
    )


def test_flow_validates_success_branch_only():
    data = textwrap.dedent(
        """
        flows:
            add_contact:
                description: add a contact to your contact list
                steps:
                - noop: true
                  next:
                  - if: true
                    then: END
                  - else: END
        """
    )
    assert YAMLFlowsReader.read_from_string(data)


def test_flow_validates_invalid_step_content():
    data = textwrap.dedent(
        """
        flows:
            add_contact:
                description: add a contact to your contact list
                steps:
                - foo: bar
        """
    )
    with pytest.raises(YamlValidationException) as e:
        YAMLFlowsReader.read_from_string(data)
    expected_error = (
        "Not a valid 'steps' definition. Expected action step "
        "or call step or collect step or link step "
        "or slot set step."
    )
    assert expected_error in str(e.value)


def test_flow_validates_true_flow_guard():
    data = textwrap.dedent(
        """
        flows:
            add_contact:
                if: true
                description: add a contact to your contact list
                steps:
                - noop: true
                  next: END
        """
    )
    assert YAMLFlowsReader.read_from_string(data)


def test_flow_invalidates_noop_step():
    data = textwrap.dedent(
        """
        flows:
            add_contact:
                description: add a contact to your contact list
                steps:
                - noop: true
        """
    )
    with pytest.raises(YamlValidationException):
        YAMLFlowsReader.read_from_string(data)


def test_flow_validates_missing_flow_description() -> None:
    data = textwrap.dedent(
        """
        flows:
            add_contact:
                steps:
                - noop: true
        """
    )
    with pytest.raises(YamlValidationException) as e:
        YAMLFlowsReader.read_from_string(data)
    assert "'description' is a required property" in str(e.value)


def test_read_flow_with_metadata_with_line_numbers() -> None:
    flows = YAMLFlowsReader.read_from_file("data/test_flows/flows_with_metadata.yml")

    assert len(flows.user_flows) == 2
    assert "line_numbers" in flows.user_flows.underlying_flows[0].steps[0].metadata
    assert (
        flows.user_flows.underlying_flows[0].steps[0].metadata["line_numbers"] == "5-5"
    )


def test_read_flow_without_metadata_with_line_numbers() -> None:
    flows = YAMLFlowsReader.read_from_file("data/test_flows/basic_flows.yml")
    flows_with_metadata = YAMLFlowsReader.read_from_file(
        "data/test_flows/flows_with_metadata.yml"
    )

    assert flows == flows_with_metadata
