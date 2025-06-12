from pathlib import Path
from unittest.mock import MagicMock

import pytest
from structlog.testing import capture_logs

from rasa.shared.core.domain import Domain
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.studio import data_handler
from rasa.studio.constants import (
    STUDIO_DOMAIN_FILENAME,
    STUDIO_FLOWS_FILENAME,
    STUDIO_NLU_FILENAME,
)
from rasa.studio.data_handler import StudioDataHandler
from rasa.studio.download.download import (
    _handle_download_no_overwrite,
    _handle_download_with_overwrite,
    _merge_data_no_overwrite,
    _merge_dir_data_no_overwrite,
    _merge_directory_domain,
    _merge_domain_no_overwrite,
    _merge_file_data_no_overwrite,
    _merge_file_domain,
    _prepare_data_and_domain_paths,
    handle_download,
)


@pytest.fixture
def mock_args(tmp_path: Path) -> MagicMock:
    """Fixture to create a mock argparse.Namespace with default paths."""
    args = MagicMock()
    args.domain = None  # means "use default domain path"
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir(parents=True, exist_ok=True)
    args.data = str(data_dir)
    args.overwrite = False
    args.config = None
    args.endpoints = None
    args.assistant_name = ["my_assistant"]
    return args


@pytest.fixture
def mock_studio_data_handler() -> MagicMock:
    """Fixture to create a mock StudioDataHandler with has_nlu set to False."""
    handler = MagicMock(spec=StudioDataHandler)
    handler.has_nlu.return_value = False
    return handler


def test_prepare_data_and_domain_paths_no_domain(mock_args: MagicMock):
    """Test domain and data paths are prepared correctly when no domain is provided."""
    domain_path, data_path = _prepare_data_and_domain_paths(mock_args)
    assert domain_path.exists()
    assert domain_path.is_file()
    assert data_path.is_dir()


def test_handle_download_no_overwrite_dir(
    tmp_path: Path,
    mock_args: MagicMock,
    mock_studio_data_handler: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test downloading studio data without overwriting local data."""
    # Arrange
    domain_dir = tmp_path / "domain"
    domain_dir.mkdir()

    data_from_studio = MagicMock(spec=TrainingDataImporter)
    data_local = MagicMock(spec=TrainingDataImporter)
    # Local domain is empty and studio domain has one intent
    data_local.get_user_domain.return_value = Domain.empty()
    data_from_studio.get_user_domain.return_value = Domain.from_dict(
        {"intents": ["hello"]}
    )

    mock_import_data = MagicMock(return_value=(data_from_studio, data_local))
    monkeypatch.setattr(
        "rasa.studio.download.download.import_data_from_studio", mock_import_data
    )

    mock_args.domain = str(domain_dir)
    domain_path, data_paths = _prepare_data_and_domain_paths(mock_args)

    # Act
    _handle_download_no_overwrite(mock_studio_data_handler, domain_path, data_paths)

    # Assert
    leftover_path = domain_dir / STUDIO_DOMAIN_FILENAME
    assert leftover_path.exists()

    leftover_domain = Domain.from_file(str(leftover_path))
    assert "hello" in leftover_domain.intents


def test_handle_download_with_overwrite(
    tmp_path: Path,
    mock_args: MagicMock,
    mock_studio_data_handler: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test downloading studio data with overwriting local data."""
    domain_file = tmp_path / "domain.yml"
    entity = {"entity": "name", "value": "Jane", "role": "contact", "group": "test"}
    domain = Domain.from_dict({"entities": [entity]})
    domain_file.write_text(domain.as_yaml())

    # Create a temporary data file
    data_file = tmp_path / "data.yml"
    data_file.touch()

    data_from_studio = MagicMock(spec=TrainingDataImporter)
    data_local = MagicMock(spec=TrainingDataImporter)

    # Update entity so that studio data has "John"
    entity["value"] = "John"
    data_from_studio.get_user_domain.return_value = Domain.from_dict(
        {"entities": [entity]}
    )

    mock_import_data = MagicMock(return_value=(data_from_studio, data_local))
    monkeypatch.setattr(
        "rasa.studio.download.download.import_data_from_studio", mock_import_data
    )

    _handle_download_with_overwrite(mock_studio_data_handler, domain_file, data_file)

    # Leftover domain should not exist
    leftover_path = tmp_path / STUDIO_DOMAIN_FILENAME
    assert not leftover_path.exists()

    updated_domain = Domain.from_file(str(domain_file))
    assert updated_domain.as_dict()["entities"][0]["value"] == "John"


def test_handle_download_main(mock_args: MagicMock, monkeypatch: pytest.MonkeyPatch):
    """Test the main handle_download function to ensure it calls the correct handler."""
    mock_confirm = MagicMock()
    mock_confirm.return_value.ask.return_value = True

    mock_prep = MagicMock(return_value=(Path("domain.yml"), [Path("data")]))
    mock_no_overwrite = MagicMock()
    mock_overwrite = MagicMock()

    monkeypatch.setattr(
        "rasa.studio.download.download.questionary.confirm", mock_confirm
    )
    monkeypatch.setattr(
        "rasa.studio.download.download._prepare_data_and_domain_paths", mock_prep
    )
    monkeypatch.setattr(
        "rasa.studio.download.download._handle_download_no_overwrite", mock_no_overwrite
    )
    monkeypatch.setattr(
        "rasa.studio.download.download._handle_download_with_overwrite", mock_overwrite
    )

    mock_handler = MagicMock(spec=StudioDataHandler)
    mock_handler.request_all_data.return_value = None
    mock_handler.get_config.return_value = "some config text"
    mock_handler.get_endpoints.return_value = "some endpoints text"
    mock_handler_init = MagicMock(return_value=mock_handler)

    monkeypatch.setattr(
        "rasa.studio.download.download.StudioDataHandler", mock_handler_init
    )

    # Test when overwrite is False
    mock_args.overwrite = False
    handle_download(mock_args)

    mock_no_overwrite.assert_called_once()
    mock_overwrite.assert_not_called()

    mock_no_overwrite.reset_mock()
    mock_overwrite.reset_mock()

    # Test when overwrite is True
    mock_args.overwrite = True
    handle_download(mock_args)

    mock_overwrite.assert_called_once()
    mock_no_overwrite.assert_not_called()


def test_merge_domain_no_overwrite_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    domain_path = tmp_path

    data_from_studio = MagicMock(spec=TrainingDataImporter)
    data_local = MagicMock(spec=TrainingDataImporter)

    mock_dir_merge = MagicMock()
    mock_file_merge = MagicMock()

    monkeypatch.setattr(
        "rasa.studio.download.download._merge_directory_domain", mock_dir_merge
    )

    _merge_domain_no_overwrite(domain_path, data_from_studio, data_local)

    mock_dir_merge.assert_called_once_with(domain_path, data_from_studio, data_local)
    mock_file_merge.assert_not_called()


def test_merge_domain_no_overwrite_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    domain_path = tmp_path / "domain.yml"
    domain_path.touch()

    data_from_studio = MagicMock(spec=TrainingDataImporter)
    data_local = MagicMock(spec=TrainingDataImporter)

    mock_dir_merge = MagicMock()
    mock_file_merge = MagicMock()

    monkeypatch.setattr(
        "rasa.studio.download.download._merge_file_domain", mock_file_merge
    )

    _merge_domain_no_overwrite(domain_path, data_from_studio, data_local)

    mock_file_merge.assert_called_once_with(domain_path, data_from_studio, data_local)
    mock_dir_merge.assert_not_called()


def test_merge_directory_domain(
    tmp_path: Path, caplog, monkeypatch: pytest.MonkeyPatch
):
    domain_dir = tmp_path / "domain_dir"
    domain_dir.mkdir()

    data_from_studio = MagicMock(spec=TrainingDataImporter)
    data_local = MagicMock(spec=TrainingDataImporter)

    # Mock combine_domains to produce a domain with an intent "test_intent"
    combined_domain_data = {"intents": ["test_intent"]}

    mock_combine = MagicMock(return_value=combined_domain_data)
    monkeypatch.setattr(data_handler, "combine_domains", mock_combine)

    _merge_directory_domain(domain_dir, data_from_studio, data_local)

    # Verify the new domain file was created
    new_domain_path = domain_dir / STUDIO_NLU_FILENAME
    assert new_domain_path.exists()

    persisted_domain = Domain.from_file(str(new_domain_path))
    assert "test_intent" in persisted_domain.intents


def test_merge_directory_domain_no_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_from_studio = MagicMock(spec=TrainingDataImporter)
    data_local = MagicMock(spec=TrainingDataImporter)

    mock_combine = MagicMock(return_value={})
    monkeypatch.setattr(data_handler, "combine_domains", mock_combine)

    with capture_logs() as cap_logs:
        _merge_directory_domain(tmp_path, data_from_studio, data_local)
        warning_message = "No additional domain data found in Studio assistant."
        assert warning_message in cap_logs[-1]["event_info"]

    # Since the domain was empty, no file should be created
    new_domain_path = tmp_path / STUDIO_NLU_FILENAME
    assert not new_domain_path.exists()


def test_merge_file_domain(tmp_path: Path):
    domain_file = tmp_path / "domain.yml"
    initial_domain = Domain.from_dict({"intents": ["old_intent"]})
    domain_file.write_text(initial_domain.as_yaml())

    data_from_studio = MagicMock(spec=TrainingDataImporter)
    data_local = MagicMock(spec=TrainingDataImporter)

    # Local domain is the existing file, and studio domain has "new_intent"
    local_domain = Domain.from_file(str(domain_file))
    data_local.get_user_domain.return_value = local_domain
    data_from_studio.get_user_domain.return_value = Domain.from_dict(
        {"intents": ["new_intent"]}
    )

    _merge_file_domain(domain_file, data_from_studio, data_local)

    updated_domain = Domain.from_file(str(domain_file))
    assert "old_intent" in updated_domain.intents
    assert "new_intent" in updated_domain.intents


def test_merge_data_no_overwrite_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_path = tmp_path / "data.yml"
    data_path.touch()

    handler = MagicMock()
    data_from_studio = MagicMock(spec=TrainingDataImporter)
    data_local = MagicMock(spec=TrainingDataImporter)

    mock_file = MagicMock()
    mock_dir = MagicMock()

    monkeypatch.setattr(
        "rasa.studio.download.download._merge_file_data_no_overwrite", mock_file
    )
    monkeypatch.setattr(
        "rasa.studio.download.download._merge_dir_data_no_overwrite", mock_dir
    )

    _merge_data_no_overwrite(data_path, handler, data_from_studio, data_local)

    mock_file.assert_called_once()
    mock_dir.assert_not_called()


def test_merge_data_no_overwrite_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_path = tmp_path / "data_dir"
    data_path.mkdir()

    handler = MagicMock()
    data_from_studio = MagicMock(spec=TrainingDataImporter)
    data_local = MagicMock(spec=TrainingDataImporter)

    mock_file = MagicMock()
    mock_dir = MagicMock()

    monkeypatch.setattr(
        "rasa.studio.download.download._merge_file_data_no_overwrite", mock_file
    )
    monkeypatch.setattr(
        "rasa.studio.download.download._merge_dir_data_no_overwrite", mock_dir
    )

    _merge_data_no_overwrite(data_path, handler, data_from_studio, data_local)

    mock_dir.assert_called_once()
    mock_file.assert_not_called()


def test_merge_file_data_no_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_file = tmp_path / "data.yml"
    data_file.touch()

    handler = MagicMock()
    handler.has_nlu.return_value = True
    handler.has_flows.return_value = True

    data_from_studio = MagicMock(spec=TrainingDataImporter)
    data_local = MagicMock(spec=TrainingDataImporter)

    mock_nlu = MagicMock()
    mock_flows = MagicMock()

    data_local.get_nlu_data.return_value.merge.return_value = mock_nlu
    data_local.get_user_flows.return_value.merge.return_value = mock_flows

    mock_nlu_persist = MagicMock()
    monkeypatch.setattr(mock_nlu, "persist_nlu", mock_nlu_persist)

    mock_yaml_dump = MagicMock()
    monkeypatch.setattr(
        "rasa.shared.core.flows.yaml_flows_io.YamlFlowsWriter.dump", mock_yaml_dump
    )

    _merge_file_data_no_overwrite(data_file, handler, data_from_studio, data_local)

    mock_nlu_persist.assert_called_once_with(str(data_file))
    mock_yaml_dump.assert_called_once_with(mock_flows.underlying_flows, data_file)


def test_merge_dir_data_no_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    handler = MagicMock()
    handler.has_nlu.return_value = True
    handler.has_flows.return_value = True

    data_from_studio = MagicMock(spec=TrainingDataImporter)
    data_local = MagicMock(spec=TrainingDataImporter)

    mock_nlu_diff = MagicMock()
    mock_flows_diff = MagicMock()

    monkeypatch.setattr(
        "rasa.studio.download.download._persist_nlu_diff", mock_nlu_diff
    )
    monkeypatch.setattr(
        "rasa.studio.download.download._persist_flows_diff", mock_flows_diff
    )

    _merge_dir_data_no_overwrite(tmp_path, handler, data_from_studio, data_local)

    nlu_path = tmp_path / STUDIO_NLU_FILENAME
    flows_path = tmp_path / STUDIO_FLOWS_FILENAME

    mock_nlu_diff.assert_called_once_with(data_local, data_from_studio, nlu_path)
    mock_flows_diff.assert_called_once_with(data_local, data_from_studio, flows_path)
