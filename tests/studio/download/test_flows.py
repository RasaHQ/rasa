from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rasa.shared.core.flows import Flow, FlowsList
from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence
from rasa.shared.core.flows.steps import ActionFlowStep
from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader, YamlFlowsWriter
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.studio.constants import STUDIO_FLOWS_FILENAME
from rasa.studio.download.flows import STUDIO_FLOWS_DIR_NAME, merge_flows_with_overwrite
from rasa.utils.mapper import RasaPrimitiveStorageMapper


def _write_single_file_flows(file_path: Path, flows_list: FlowsList) -> None:
    YamlFlowsWriter.dump(flows_list.underlying_flows, file_path)


def _read_single_file_flows(file_path: Path) -> FlowsList:
    return YAMLFlowsReader.read_from_file(file_path, add_line_numbers=False)


@pytest.fixture
def mock_studio_data_handler() -> MagicMock:
    handler = MagicMock()
    handler.has_nlu.return_value = False
    handler.has_flows.return_value = True
    return handler


@pytest.fixture
def mock_studio_data_importer() -> MagicMock:
    return MagicMock(spec=TrainingDataImporter)


@pytest.fixture
def mock_local_data_importer() -> MagicMock:
    return MagicMock(spec=TrainingDataImporter)


@pytest.fixture
def flow_step_sequence() -> FlowStepSequence:
    step = ActionFlowStep(
        idx=1,
        action="action_listen",
        custom_id="my_step",
        description=None,
        metadata={},
        next=FlowStepLinks(links=[]),
        flow_id="foo",
    )
    return FlowStepSequence(child_steps=[step])


def test_merge_flows_with_overwrite_file(
    tmp_path: Path,
    mock_studio_data_handler: MagicMock,
    mock_studio_data_importer: MagicMock,
    mock_local_data_importer: MagicMock,
):
    """If data_path is a file, we do a full merge of studio and local data."""
    # Create a local data file with flow_A
    flow_a = Flow(id="flow_A", description="Flow A")
    local_flows_list = FlowsList([flow_a])
    local_file = tmp_path / "training_data.yml"
    _write_single_file_flows(local_file, local_flows_list)
    mock_local_data_importer.get_user_flows.return_value = local_flows_list

    # Create a studio data file with flow_B
    flow_b = Flow(id="flow_B", description="Flow B")
    studio_flows_list = FlowsList([flow_a, flow_b])
    mock_studio_data_importer.get_user_flows.return_value = studio_flows_list

    data_paths = [local_file]
    mapper = RasaPrimitiveStorageMapper(
        domain_path=None, training_data_paths=data_paths
    )

    merge_flows_with_overwrite(
        data_paths=data_paths,
        handler=mock_studio_data_handler,
        data_from_studio=mock_studio_data_importer,
        data_local=mock_local_data_importer,
        mapper=mapper,
    )

    # Now local_file should contain both flow_A and flow_B
    updated_flows = _read_single_file_flows(local_file)
    assert len(updated_flows.underlying_flows) == 2
    assert updated_flows.flow_by_id("flow_A") is not None
    assert updated_flows.flow_by_id("flow_B") is not None


def test_merge_flows_with_overwrite_dir(
    tmp_path: Path,
    mock_studio_data_handler: MagicMock,
    mock_studio_data_importer: MagicMock,
    mock_local_data_importer: MagicMock,
    flow_step_sequence: FlowStepSequence,
):
    """If data_path is a directory, we do a partial merge of studio and local data."""
    data_dir = tmp_path / "training_data_dir"
    data_dir.mkdir()

    # Create a local data file with flow_X
    flow_x = Flow(id="flow_X", description="Flow X", step_sequence=flow_step_sequence)
    local_flows_list = FlowsList([flow_x])
    local_file = data_dir / "local_flows.yml"
    _write_single_file_flows(local_file, local_flows_list)
    mock_local_data_importer.get_user_flows.return_value = local_flows_list

    # Create a studio data file with flow_X (shared) + flow_Y (new)
    flow_y = Flow(id="flow_Y", description="Flow Y")
    studio_flows_list = FlowsList([flow_x, flow_y])
    mock_studio_data_importer.get_user_flows.return_value = studio_flows_list

    data_paths = [data_dir]
    mapper = RasaPrimitiveStorageMapper(None, data_paths)

    merge_flows_with_overwrite(
        data_paths=data_paths,
        handler=mock_studio_data_handler,
        data_from_studio=mock_studio_data_importer,
        data_local=mock_local_data_importer,
        mapper=mapper,
    )

    # local_flows.yml should have flow_X only
    updated_local = _read_single_file_flows(local_file)
    assert len(updated_local) == 1
    assert updated_local.flow_by_id("flow_X")

    # New flow, flow_Y, goes to studio_flows/flow_Y.yml
    studio_flows_dir = data_dir / STUDIO_FLOWS_DIR_NAME
    assert studio_flows_dir.is_dir()
    flow_y_file = studio_flows_dir / "flow_Y.yml"
    assert flow_y_file.exists()

    # Check that the flow was written correctly
    new_flows_y = _read_single_file_flows(flow_y_file)
    assert new_flows_y.flow_by_id("flow_Y") is not None


def test_merge_flows_with_overwrite_dir_no_leftover(
    tmp_path: Path,
    mock_studio_data_handler: MagicMock,
    mock_studio_data_importer: MagicMock,
    mock_local_data_importer: MagicMock,
    flow_step_sequence: FlowStepSequence,
):
    """If local and studio flows match, there should be no leftover flows."""
    data_dir = tmp_path / "my_flows_dir"
    data_dir.mkdir()

    # Create flow_A to be used in both local and studio data
    flow_a = Flow(id="flow_A", description="Flow A", step_sequence=flow_step_sequence)
    local_flows = FlowsList([flow_a])
    local_file = data_dir / "local_flows.yml"
    _write_single_file_flows(local_file, local_flows)

    mock_local_data_importer.get_user_flows.return_value = local_flows
    mock_studio_data_importer.get_user_flows.return_value = local_flows

    data_paths = [data_dir]
    mapper = RasaPrimitiveStorageMapper(None, data_paths)

    merge_flows_with_overwrite(
        data_paths=data_paths,
        handler=mock_studio_data_handler,
        data_from_studio=mock_studio_data_importer,
        data_local=mock_local_data_importer,
        mapper=mapper,
    )

    # New directory with leftover studio flows should not be created
    studio_flows_dir = data_dir / STUDIO_FLOWS_FILENAME
    assert not studio_flows_dir.exists()


def test_merge_flows_with_overwrite_dir_has_leftover(
    tmp_path: Path,
    mock_studio_data_handler: MagicMock,
    mock_studio_data_importer: MagicMock,
    mock_local_data_importer: MagicMock,
    flow_step_sequence: FlowStepSequence,
):
    """If local and studio flows differ, there should be a leftover flows directory."""
    data_dir = tmp_path / "flows_data_dir_leftover"
    data_dir.mkdir()

    # Create a local data file with flow_one and flow_two (shared)
    local_flow_one = Flow(
        id="flow_one", description="Flow one", step_sequence=flow_step_sequence
    )
    local_flow_two = Flow(
        id="flow_two", description="Flow two", step_sequence=flow_step_sequence
    )
    local_file = data_dir / "my_local_flows.yml"
    local_flows_list = FlowsList([local_flow_one, local_flow_two])
    _write_single_file_flows(local_file, local_flows_list)
    mock_local_data_importer.get_user_flows.return_value = local_flows_list

    # Create a studio data file with flow_two (shared) and flow_three and flow_four
    flow_three = Flow(id="flow_three", description="Flow three")
    flow_four = Flow(id="flow_four", description="Flow four")
    studio_flows_list = FlowsList([local_flow_two, flow_three, flow_four])
    mock_studio_data_importer.get_user_flows.return_value = studio_flows_list

    data_paths = [data_dir]
    mapper = RasaPrimitiveStorageMapper(None, data_paths)

    merge_flows_with_overwrite(
        data_paths=data_paths,
        handler=mock_studio_data_handler,
        data_from_studio=mock_studio_data_importer,
        data_local=mock_local_data_importer,
        mapper=mapper,
    )

    # local_flows.yml should have flow_one and flow_two
    updated_local = _read_single_file_flows(local_file)
    assert len(updated_local) == 2
    assert updated_local.flow_by_id("flow_one")
    assert updated_local.flow_by_id("flow_two")

    # New flows go to the "studio_flows" directory
    studio_subdir = data_dir / STUDIO_FLOWS_DIR_NAME
    assert studio_subdir.is_dir()

    # The "studio_flows" directory should contain flow_three.yml and flow_four.yml
    flow_three_file = studio_subdir / "flow_three.yml"
    flow_four_file = studio_subdir / "flow_four.yml"
    assert flow_three_file.exists()
    assert flow_four_file.exists()

    # Check that the flows were written correctly
    read_three = _read_single_file_flows(flow_three_file)
    read_four = _read_single_file_flows(flow_four_file)
    assert read_three.flow_by_id("flow_three") is not None
    assert read_four.flow_by_id("flow_four") is not None
