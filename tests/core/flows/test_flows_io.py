import os
import pytest
import tempfile
from rasa.shared.core.flows.yaml_flows_io import (
    is_flows_file,
    YAMLFlowsReader,
    YamlFlowsWriter,
)


@pytest.fixture(scope="module")
def basic_flows_file(tests_data_folder: str) -> str:
    return os.path.join(tests_data_folder, "test_flows", "basic_flows.yml")


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

    re_read_flows_list = YAMLFlowsReader.read_from_file(tmp_file_name)
    assert re_read_flows_list.as_dict() == flows_list.as_dict()
