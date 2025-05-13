import os
import tempfile
import textwrap
from dataclasses import MISSING, fields

import pytest
import yaml

from rasa.shared.core.flows.constants import KEY_TRANSLATION
from rasa.shared.core.flows.flow import Flow, FlowLanguageTranslation
from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence
from rasa.shared.core.flows.steps.collect import CollectInformationFlowStep
from rasa.shared.core.flows.yaml_flows_io import (
    YAMLFlowsReader,
    YamlFlowsWriter,
    get_flow_as_json,
    is_flows_file,
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
    assert (
        flows.underlying_flows[0].file_path == "data/test_flows/flows_with_metadata.yml"
    )


def test_read_flow_without_metadata_with_line_numbers() -> None:
    flows = YAMLFlowsReader.read_from_file("data/test_flows/basic_flows.yml")
    flows_with_metadata = YAMLFlowsReader.read_from_file(
        "data/test_flows/flows_with_metadata.yml"
    )

    for flow in flows.underlying_flows:
        assert flow.file_path == "data/test_flows/basic_flows.yml"
        flow.file_path = None
    for flow in flows_with_metadata.underlying_flows:
        assert flow.file_path == "data/test_flows/flows_with_metadata.yml"
        flow.file_path = None

    assert flows == flows_with_metadata


def test_read_flow_with_name_translation() -> None:
    # Define flows and their translations as dictionaries.
    flows_definition = {
        "add_contact": {
            "name": "add a contact",
            "description": "Flow to add a contact to your contact list",
            "translation": {
                "en": {"name": "Add a Contact"},
                "it": {"name": "Aggiungi un Contatto"},
                "de": {"name": "Kontakt hinzufügen"},
            },
            "steps": [{"action": "utter_add_contact"}],
        },
        "remove_contact": {
            "name": "remove a contact",
            "description": "Another test flow",
            "translation": {
                "en": {"name": "Remove a Contact"},
                "it": {"name": "Rimuovi un Contatto"},
                "de": {"name": "Kontakt entfernen"},
            },
            "steps": [{"action": "utter_remove_contact"}],
        },
    }

    # Dump the flows_definition into a YAML string.
    data = yaml.dump({"flows": flows_definition}, sort_keys=False)

    # Read the flows from the generated YAML string.
    flows = YAMLFlowsReader.read_from_string(data, add_line_numbers=False)

    # Compare the expected translations with those parsed from YAML.
    for flow in flows.underlying_flows:
        expected_translations = {
            language: FlowLanguageTranslation.parse_obj(data)
            for language, data in flows_definition[flow.id][KEY_TRANSLATION].items()
        }
        assert flow.translation == expected_translations


def test_read_flow_with_missing_translation() -> None:
    flows_definition = {
        "add_contact": {
            "name": "add a contact",
            "description": "Flow to add a contact to your contact list",
            "steps": [{"action": "utter_add_contact"}],
        }
    }
    data = yaml.dump({"flows": flows_definition}, sort_keys=False)
    flows = YAMLFlowsReader.read_from_string(data, add_line_numbers=False)
    flow = flows.underlying_flows[0]
    assert flow.translation == {}


def test_read_flow_with_empty_translation() -> None:
    flows_definition = {
        "add_contact": {
            "name": "add a contact",
            "description": "Flow to add a contact to your contact list",
            "translation": {},
            "steps": [{"action": "utter_add_contact"}],
        }
    }
    data = yaml.dump({"flows": flows_definition}, sort_keys=False)
    flows = YAMLFlowsReader.read_from_string(data, add_line_numbers=False)
    flow = flows.underlying_flows[0]
    assert flow.translation == {}


def test_read_flow_with_invalid_translation_format() -> None:
    flows_definition = {
        "add_contact": {
            "name": "add a contact",
            "description": "Flow to add a contact to your contact list",
            "translation": "invalid_format",
            "steps": [{"action": "utter_add_contact"}],
        }
    }
    data = yaml.dump({"flows": flows_definition}, sort_keys=False)
    with pytest.raises(YamlValidationException):
        YAMLFlowsReader.read_from_string(data, add_line_numbers=False)


def test_get_flow_as_json_removes_defaults():
    # Create a Flow with default fields
    flow = Flow(
        id="test_flow",
        run_pattern_completed=True,  # default
        persisted_slots=[],
        file_path="some/path/to_flow.yml",  # always removed
        step_sequence=FlowStepSequence(
            [
                CollectInformationFlowStep(
                    custom_id=None,
                    idx=0,
                    description=None,
                    metadata={},
                    next=FlowStepLinks([]),
                    flow_id="test_flow",
                    collect="amount",
                    utter="utter_ask_amount",  # default
                    collect_action="action_ask_amount",
                    rejections=[],  # default
                    ask_before_filling=False,  # default
                    reset_after_flow_ends=True,  # default
                    force_slot_filling=False,  # default
                )
            ]
        ),
    )

    # Generate JSON with and without cleaning
    uncleaned = get_flow_as_json(flow, should_clean_json=False)
    cleaned = get_flow_as_json(flow, should_clean_json=True)

    assert "run_pattern_completed" in uncleaned
    assert "run_pattern_completed" not in cleaned

    assert "file_path" in uncleaned
    assert "file_path" not in cleaned

    # Check the first step differences
    uncleaned_step = uncleaned["steps"][0]
    cleaned_step = cleaned["steps"][0]

    assert uncleaned_step.get("utter") == "utter_ask_amount"
    assert "utter" not in cleaned_step

    assert uncleaned_step.get("ask_before_filling") is False
    assert "ask_before_filling" not in cleaned_step

    assert uncleaned_step.get("reset_after_flow_ends") is True
    assert "reset_after_flow_ends" not in cleaned_step

    assert uncleaned_step.get("force_slot_filling") is False
    assert "force_slot_filling" not in cleaned_step

    assert uncleaned_step.get("rejections") == []
    assert "rejections" not in cleaned_step

    assert "id" in uncleaned_step
    assert "id" not in cleaned_step


def test_collectinformationflowstep_defaults_cleaned_from_json():
    """
    Test if adding a new default field on CollectInformationFlowStep
    is addressed in get_flow_as_json.
    """
    # Initialize the step
    step_data = {"collect": "my_slot"}
    step = CollectInformationFlowStep.from_json(flow_id="test_flow", data=step_data)
    step_json = step.as_json()

    # Initialize the flow
    flow = Flow(
        id="test_flow",
        step_sequence=FlowStepSequence([step]),
    )

    # Dump JSON with and without cleaning
    uncleaned_flow_json = get_flow_as_json(flow, should_clean_json=False)
    cleaned_flow_json = get_flow_as_json(flow, should_clean_json=True)

    uncleaned_step_data = uncleaned_flow_json["steps"][0]
    cleaned_step_data = cleaned_flow_json["steps"][0]

    # The uncleaned step data should match what the step itself produces.
    assert uncleaned_step_data == step_json

    # For each field that has a default or default_factory, ensure it's not
    # present as that default value in the cleaned step JSON.
    for field_info in fields(CollectInformationFlowStep):
        has_default = field_info.default is not MISSING
        has_default_factory = (
            getattr(field_info, "default_factory", MISSING) is not MISSING
        )

        if has_default or has_default_factory:
            field_name = field_info.name
            # If the default field is dumped in cleaned JSON
            if field_name in cleaned_step_data:
                # Figure out the default value
                if has_default:
                    default_val = field_info.default
                else:
                    default_val = field_info.default_factory()

                assert cleaned_step_data[field_name] != default_val, (
                    f"Field '{field_name}' remains in cleaned JSON with the default "
                    f"value '{default_val}'. Update clean logic to remove it."
                )

    # Confirm a known default is definitely removed in cleaned
    assert "utter" in uncleaned_step_data
    assert "utter" not in cleaned_step_data
