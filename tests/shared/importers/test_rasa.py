import os
from pathlib import Path
from typing import Text
from unittest.mock import MagicMock

from _pytest.monkeypatch import MonkeyPatch

from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_CONVERSATION_TEST_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
)
from rasa.shared.core.constants import (
    DEFAULT_ACTION_NAMES,
    DEFAULT_INTENTS,
    DEFAULT_SLOT_NAMES,
    REQUESTED_SLOT,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.slots import AnySlot
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.importers.rasa import RasaFileImporter


def test_rasa_file_importer(project: Text):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = os.path.join(project, DEFAULT_DOMAIN_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)

    importer = RasaFileImporter(config_path, domain_path, [default_data_path])

    domain = importer.get_domain()
    assert len(domain.intents) == 7 + len(DEFAULT_INTENTS)
    default_slots = [
        AnySlot(slot_name, mappings=[{}])
        for slot_name in DEFAULT_SLOT_NAMES
        if slot_name != REQUESTED_SLOT
    ]
    assert sorted(domain.slots, key=lambda s: s.name) == sorted(
        default_slots, key=lambda s: s.name
    )

    assert domain.entities == []
    assert len(domain.action_names_or_texts) == 6 + len(DEFAULT_ACTION_NAMES)
    assert len(domain.responses) == 6

    stories = importer.get_stories()
    assert len(stories.story_steps) == 5

    test_stories = importer.get_conversation_tests()
    assert len(test_stories.story_steps) == 0

    nlu_data = importer.get_nlu_data("en")
    assert len(nlu_data.intents) == 7
    assert len(nlu_data.intent_examples) == 68


def test_read_conversation_tests(project: Text):
    importer = RasaFileImporter(
        training_data_paths=[str(Path(project) / DEFAULT_CONVERSATION_TEST_PATH)]
    )

    test_stories = importer.get_conversation_tests()
    assert len(test_stories.story_steps) == 7


def test_rasa_file_importer_with_invalid_config():
    importer = RasaFileImporter(config_file="invalid path")
    actual = importer.get_config()

    assert actual == {}


def test_rasa_file_importer_with_invalid_domain(tmp_path: Path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("")
    importer = TrainingDataImporter.load_from_dict({}, str(config_file), None, [])

    actual = importer.get_domain()
    assert actual.as_dict() == Domain.empty().as_dict()


def test_rasa_file_importer_cached_get_config(
    monkeypatch: MonkeyPatch,
    empty_config_file: Path,
    small_domain_file: Path,
) -> None:
    """Test that the cached result of get_config is used."""

    mock_read_model_configuration = MagicMock()
    monkeypatch.setattr(
        "rasa.shared.importers.rasa.read_model_configuration",
        mock_read_model_configuration,
    )

    importer = RasaFileImporter(
        config_file=str(empty_config_file), domain_path=str(small_domain_file)
    )
    importer.get_config()

    assert mock_read_model_configuration.call_count == 1

    importer.get_config()

    # the fact that mock of read_model_configuration was only called once
    # indicates that the cached result was used for get_config
    assert mock_read_model_configuration.call_count == 1


def test_rasa_file_importer_cached_get_stories(
    monkeypatch: MonkeyPatch,
    empty_config_file: Path,
    small_domain_file: Path,
) -> None:
    """Test that the cached result of get_stories is used."""

    mock_story_graph_from_paths = MagicMock()
    monkeypatch.setattr(
        "rasa.shared.importers.rasa.utils.story_graph_from_paths",
        mock_story_graph_from_paths,
    )

    importer = RasaFileImporter(
        config_file=str(empty_config_file), domain_path=str(small_domain_file)
    )
    importer.get_stories()

    assert mock_story_graph_from_paths.call_count == 1

    importer.get_stories()

    # the fact that mock of read_model_configuration was only called once
    # indicates that the cached result was used for get_stories
    assert mock_story_graph_from_paths.call_count == 1


def test_rasa_file_importer_cached_get_flows(
    monkeypatch: MonkeyPatch,
    empty_config_file: Path,
    small_domain_file: Path,
) -> None:
    """Test that the cached result of get_flows is used."""

    mock_story_graph_from_paths = MagicMock()
    monkeypatch.setattr(
        "rasa.shared.importers.rasa.utils.flows_from_paths",
        mock_story_graph_from_paths,
    )

    importer = RasaFileImporter(
        config_file=str(empty_config_file), domain_path=str(small_domain_file)
    )
    importer.get_flows()

    assert mock_story_graph_from_paths.call_count == 1

    importer.get_flows()

    # the fact that mock of read_model_configuration was only called once
    # indicates that the cached result was used for get_flows
    assert mock_story_graph_from_paths.call_count == 1


def test_rasa_file_importer_cached_get_conversation_tests(
    monkeypatch: MonkeyPatch,
    empty_config_file: Path,
    small_domain_file: Path,
) -> None:
    """Test that the cached result of get_conversation_tests is used."""

    mock_story_graph_from_paths = MagicMock()
    monkeypatch.setattr(
        "rasa.shared.importers.rasa.utils.story_graph_from_paths",
        mock_story_graph_from_paths,
    )

    importer = RasaFileImporter(
        config_file=str(empty_config_file), domain_path=str(small_domain_file)
    )
    importer.get_conversation_tests()

    assert mock_story_graph_from_paths.call_count == 1

    importer.get_conversation_tests()

    # the fact that mock of story_graph_from_paths was only called once
    # indicates that the cached result was used for get_conversation_tests
    assert mock_story_graph_from_paths.call_count == 1


def test_rasa_file_importer_cached_get_nlu_data(
    monkeypatch: MonkeyPatch,
    empty_config_file: Path,
    small_domain_file: Path,
) -> None:
    """Test that the cached result of get_nlu_data is used."""

    mock_training_data_from_paths = MagicMock()
    monkeypatch.setattr(
        "rasa.shared.importers.rasa.utils.training_data_from_paths",
        mock_training_data_from_paths,
    )

    importer = RasaFileImporter(
        config_file=str(empty_config_file), domain_path=str(small_domain_file)
    )
    importer.get_nlu_data()

    assert mock_training_data_from_paths.call_count == 1

    importer.get_nlu_data()

    # the fact that mock of training_data_from_paths was only called once
    # indicates that the cached result was used for get_nlu_data
    assert mock_training_data_from_paths.call_count == 1


def test_rasa_file_importer_cached_get_domain(
    monkeypatch: MonkeyPatch,
    empty_config_file: Path,
    small_domain_file: Path,
) -> None:
    """Test that the cached result of get_domain is used."""
    mock_domain_load = MagicMock()
    mock_domain_load.return_value = Domain.load(small_domain_file)

    mock_empty_domain = MagicMock()
    mock_empty_domain.return_value = Domain.empty()

    monkeypatch.setattr(Domain, "load", mock_domain_load)
    monkeypatch.setattr(Domain, "empty", mock_empty_domain)

    importer = RasaFileImporter(
        config_file=str(empty_config_file), domain_path=str(small_domain_file)
    )
    importer.get_domain()

    assert mock_empty_domain.call_count == 1
    assert mock_domain_load.call_count == 1

    importer.get_domain()

    # the fact that mock of training_data_from_paths was only called once
    # indicates that the cached result was used for get_domain
    assert mock_empty_domain.call_count == 1
    assert mock_domain_load.call_count == 1
