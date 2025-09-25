from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rasa.shared.core.domain import Domain
from rasa.studio.constants import STUDIO_DOMAIN_FILENAME
from rasa.studio.pull.domains import merge_domain


@pytest.fixture
def simple_domain() -> Domain:
    """Returns a simple domain with a single intent and entity."""
    return Domain.from_dict(
        {
            "intents": ["greet"],
            "entities": ["name"],
        }
    )


@pytest.fixture
def bigger_domain() -> Domain:
    """Returns a larger domain with extra items."""
    return Domain.from_dict(
        {
            "intents": ["greet", "goodbye"],
            "entities": ["name", "date"],
            "actions": ["action_hello"],
        }
    )


@pytest.fixture
def mock_data_importer() -> MagicMock:
    """Returns a mock TrainingDataImporter that can
    return a desired 'user domain' and 'domain files'.
    """
    importer = MagicMock()
    importer.get_user_domain = MagicMock()
    importer.get_domain_files = MagicMock()
    return importer


def test_merge_domain_file(
    tmp_path: Path,
    mock_data_importer: MagicMock,
    simple_domain: Domain,
    bigger_domain: Domain,
):
    """If domain_path is a file, we do a partial merge with the local domain."""
    domain_file = tmp_path / "domain.yml"
    domain_file.write_text(simple_domain.as_yaml())

    data_from_studio = MagicMock()
    data_from_studio.get_user_domain.return_value = bigger_domain

    merge_domain(
        data_from_studio=data_from_studio,
        data_local=mock_data_importer,
        domain_path=domain_file,
    )
    merged_domain = Domain.from_file(str(domain_file))

    # `greet` stays in the local domain, `goodbye` goes to the studio domain file
    assert {"greet"}.issubset(merged_domain.intents)
    assert "goodbye" not in merged_domain.intents

    # `name` stays in the local domain, `date` goes to the studio domain file
    assert {"name"}.issubset(set(merged_domain.entities))
    assert "date" not in merged_domain.entities

    # `action_hello` is not in the local domain, so it goes to the studio domain file
    assert "action_hello" not in merged_domain.action_names_or_texts


def test_merge_domain_file_no_leftover(
    tmp_path: Path, mock_data_importer: MagicMock, simple_domain: Domain
):
    """If local and studio domains match, there should be no leftover domain."""
    domain_file = tmp_path / "domain.yml"
    domain_file.write_text(simple_domain.as_yaml())

    data_from_studio = MagicMock()
    data_from_studio.get_user_domain.return_value = simple_domain

    merge_domain(
        data_from_studio=data_from_studio,
        data_local=mock_data_importer,
        domain_path=domain_file,
    )

    leftover_path = tmp_path / STUDIO_DOMAIN_FILENAME
    assert not leftover_path.exists()


def test_merge_domain_file_has_leftover(
    tmp_path: Path,
    mock_data_importer: MagicMock,
    simple_domain: Domain,
    bigger_domain: Domain,
):
    """If local and studio domains differ, there should be a leftover domain."""
    domain_file = tmp_path / "domain.yml"
    domain_file.write_text(simple_domain.as_yaml())

    data_from_studio = MagicMock()
    data_from_studio.get_user_domain.return_value = bigger_domain

    merge_domain(
        data_from_studio=data_from_studio,
        data_local=mock_data_importer,
        domain_path=domain_file,
    )
    merged_domain = Domain.from_file(str(domain_file))
    leftover_path = tmp_path / STUDIO_DOMAIN_FILENAME
    assert leftover_path.exists()

    leftover_domain = Domain.from_file(str(leftover_path))
    # `greet` stays in the local domain, `goodbye` goes to the studio domain file
    assert "goodbye" not in merged_domain.intents
    assert "goodbye" in leftover_domain.intents

    # `name` stays in the local domain, `date` goes to the studio domain file
    assert "date" not in merged_domain.entities
    assert "date" in leftover_domain.entities

    # `action_hello` is not in the local domain, so it goes to the studio domain file
    assert "action_hello" not in merged_domain.action_names_or_texts
    assert "action_hello" in leftover_domain.action_names_or_texts


def test_merge_domain_dir(
    tmp_path: Path,
    mock_data_importer: MagicMock,
    simple_domain: Domain,
    bigger_domain: Domain,
):
    """If domain_path is a directory, we do a partial merge with the files inside."""
    domain_dir = tmp_path / "domain_dir"
    domain_dir.mkdir(parents=True, exist_ok=True)
    local_domain_file = domain_dir / "local_domain.yml"
    local_domain_file.write_text(simple_domain.as_yaml())

    mock_data_importer.get_user_domain.return_value = simple_domain
    mock_data_importer.get_domain_files.return_value = [local_domain_file]

    studio_importer = MagicMock()
    studio_importer.get_user_domain.return_value = bigger_domain

    merge_domain(
        data_from_studio=studio_importer,
        data_local=mock_data_importer,
        domain_path=domain_dir,
    )

    merged_domain = Domain.from_file(str(local_domain_file))
    # `greet` stays in the local domain, `goodbye` goes to the studio domain file
    assert {"greet"}.issubset(merged_domain.intents)
    assert "goodbye" not in merged_domain.intents

    # `name` stays in the local domain, `date` goes to the studio domain file
    assert {"name"}.issubset(set(merged_domain.entities))
    assert "date" not in merged_domain.entities

    # `action_hello` is not in the local domain, so it goes to the studio domain file
    assert "action_hello" not in merged_domain.action_names_or_texts


def test_merge_domain_dir_no_leftover(
    tmp_path: Path,
    mock_data_importer: MagicMock,
    simple_domain: Domain,
):
    """If local and studio domains match, there should be no leftover domain."""
    domain_dir = tmp_path / "domain_dir"
    domain_dir.mkdir(parents=True, exist_ok=True)
    local_domain_file = domain_dir / "local_domain.yml"
    local_domain_file.write_text(simple_domain.as_yaml())

    mock_data_importer.get_user_domain.return_value = simple_domain
    mock_data_importer.get_domain_files.return_value = [local_domain_file]

    studio_importer = MagicMock()
    studio_importer.get_user_domain.return_value = simple_domain

    merge_domain(
        data_from_studio=studio_importer,
        data_local=mock_data_importer,
        domain_path=domain_dir,
    )

    leftover_path = domain_dir / STUDIO_DOMAIN_FILENAME
    assert not leftover_path.exists()


def test_merge_domain_dir_has_leftover(
    tmp_path: Path,
    mock_data_importer: MagicMock,
    simple_domain: Domain,
    bigger_domain: Domain,
):
    """If local and studio domains differ, there should be a leftover domain."""
    domain_dir = tmp_path / "domain_dir"
    domain_dir.mkdir(parents=True, exist_ok=True)
    local_domain_file = domain_dir / "local_domain.yml"
    local_domain_file.write_text(simple_domain.as_yaml())

    mock_data_importer.get_user_domain.return_value = simple_domain
    mock_data_importer.get_domain_files.return_value = [local_domain_file]

    studio_importer = MagicMock()
    studio_importer.get_user_domain.return_value = bigger_domain

    merge_domain(
        data_from_studio=studio_importer,
        data_local=mock_data_importer,
        domain_path=domain_dir,
    )

    leftover_path = domain_dir / STUDIO_DOMAIN_FILENAME
    assert leftover_path.exists()

    merged_domain = Domain.from_file(str(local_domain_file))
    leftover_domain = Domain.from_file(str(leftover_path))
    # `greet` stays in the local domain, `goodbye` goes to the studio domain file
    assert "goodbye" not in merged_domain.intents
    assert "goodbye" in leftover_domain.intents

    # `name` stays in the local domain, `date` goes to the studio domain file
    assert "date" not in merged_domain.entities
    assert "date" in leftover_domain.entities

    # `action_hello` is not in the local domain, so it goes to the studio domain file
    assert "action_hello" not in merged_domain.action_names_or_texts
    assert "action_hello" in leftover_domain.action_names_or_texts


def test_merge_domain_dir_excludes_existing_studio_domain_file(
    tmp_path: Path,
    mock_data_importer: MagicMock,
    simple_domain: Domain,
    bigger_domain: Domain,
):
    """Test that existing studio domain file is excluded from merge."""
    domain_dir = tmp_path / "domain_dir"
    domain_dir.mkdir(parents=True, exist_ok=True)
    local_domain_file = domain_dir / "local_domain.yml"
    local_domain_file.write_text(simple_domain.as_yaml())

    # Create an existing studio domain file with some content
    existing_studio_domain_file = domain_dir / STUDIO_DOMAIN_FILENAME
    existing_studio_domain = Domain.from_dict(
        {
            "intents": ["old_intent"],
            "entities": ["old_entity"],
        }
    )
    existing_studio_domain_file.write_text(existing_studio_domain.as_yaml())

    # Mock the local data importer to return both the local domain file and the
    # existing studio domain file
    mock_data_importer.get_user_domain.return_value = simple_domain
    mock_data_importer.get_domain_files.return_value = [
        str(local_domain_file),
        str(existing_studio_domain_file),
    ]

    studio_importer = MagicMock()
    studio_importer.get_user_domain.return_value = bigger_domain

    merge_domain(
        data_from_studio=studio_importer,
        data_local=mock_data_importer,
        domain_path=domain_dir,
    )

    # Verify that the local domain file was updated correctly
    merged_domain = Domain.from_file(str(local_domain_file))
    assert {"greet"}.issubset(merged_domain.intents)
    assert "goodbye" not in merged_domain.intents
    assert {"name"}.issubset(set(merged_domain.entities))
    assert "date" not in merged_domain.entities

    # Verify that the studio domain file was overwritten with leftover items
    # (not merged with old content)
    leftover_domain = Domain.from_file(str(existing_studio_domain_file))
    assert "goodbye" in leftover_domain.intents
    assert "date" in leftover_domain.entities
    assert "action_hello" in leftover_domain.action_names_or_texts

    # Verify that the old content from the existing studio domain file is not present
    assert "old_intent" not in leftover_domain.intents
    assert "old_entity" not in leftover_domain.entities
