import shutil
from pathlib import Path
from typing import Text, Set
from unittest.mock import Mock

import pytest
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from rasa.shared.constants import CONFIG_AUTOCONFIGURABLE_KEYS
import rasa.shared.utils.io
from rasa.shared.importers import autoconfig

CONFIG_FOLDER = Path("data/test_config")

SOME_CONFIG = CONFIG_FOLDER / "stack_config.yml"
DEFAULT_CONFIG = Path("rasa/shared/importers/default_config.yml")


@pytest.mark.parametrize(
    "config_path, autoconfig_keys",
    [
        (Path("rasa/cli/initial_project/config.yml"), {"pipeline", "policies"}),
        (CONFIG_FOLDER / "config_policies_empty.yml", {"policies"}),
        (CONFIG_FOLDER / "config_pipeline_empty.yml", {"pipeline"}),
        (CONFIG_FOLDER / "config_policies_missing.yml", {"policies"}),
        (CONFIG_FOLDER / "config_pipeline_missing.yml", {"pipeline"}),
        (SOME_CONFIG, set()),
    ],
)
def test_get_configuration(
    config_path: Path, autoconfig_keys: Set[Text], monkeypatch: MonkeyPatch
):
    def _auto_configure(_, keys_to_configure: Set[Text]) -> Set[Text]:
        return keys_to_configure

    monkeypatch.setattr(autoconfig, "_dump_config", Mock())
    monkeypatch.setattr(autoconfig, "_auto_configure", _auto_configure)

    config = autoconfig.get_configuration(str(config_path))

    if autoconfig_keys:
        expected_config = _auto_configure(config, autoconfig_keys)
    else:
        expected_config = config

    assert sorted(config) == sorted(expected_config)


@pytest.mark.parametrize("config_file", ("non_existent_config.yml", None))
def test_get_configuration_missing_file(tmp_path: Path, config_file: Text):
    if config_file:
        config_file = tmp_path / "non_existent_config.yml"

    config = autoconfig.get_configuration(str(config_file))

    assert config == {}


@pytest.mark.parametrize(
    "language, keys_to_configure",
    [
        ("en", {"policies"}),
        ("en", {"pipeline"}),
        ("fr", {"pipeline"}),
        ("en", {"policies", "pipeline"}),
    ],
)
def test_auto_configure(language: Text, keys_to_configure: Set[Text]):
    expected_config = rasa.shared.utils.io.read_config_file(DEFAULT_CONFIG)

    config = autoconfig._auto_configure({"language": language}, keys_to_configure)

    for k in keys_to_configure:
        assert config[k] == expected_config[k]  # given keys are configured correctly

    assert config.get("language") == language
    config.pop("language")
    assert len(config) == len(keys_to_configure)  # no other keys are configured


@pytest.mark.parametrize(
    "config_path, missing_keys",
    [
        (CONFIG_FOLDER / "config_language_only.yml", {"pipeline", "policies"}),
        (CONFIG_FOLDER / "config_policies_missing.yml", {"policies"}),
        (CONFIG_FOLDER / "config_pipeline_missing.yml", {"pipeline"}),
        (SOME_CONFIG, []),
    ],
)
def test_add_missing_config_keys_to_file(
    tmp_path: Path, config_path: Path, missing_keys: Set[Text]
):
    config_file = str(tmp_path / "config.yml")
    shutil.copyfile(str(config_path), config_file)

    autoconfig._add_missing_config_keys_to_file(config_file, missing_keys)

    config_after_addition = rasa.shared.utils.io.read_config_file(config_file)

    assert all(key in config_after_addition for key in missing_keys)


def test_dump_config_missing_file(tmp_path: Path, capsys: CaptureFixture):

    config_path = tmp_path / "non_existent_config.yml"

    config = rasa.shared.utils.io.read_config_file(str(SOME_CONFIG))

    autoconfig._dump_config(config, str(config_path), set(), {"policies"})

    assert not config_path.exists()

    captured = capsys.readouterr()
    assert "has been removed or modified" in captured.out


# Test a few cases that are known to be potentially tricky (have failed in the past)
@pytest.mark.parametrize(
    "input_file, expected_file, autoconfig_keys",
    [
        (
            "config_with_comments.yml",
            "config_with_comments_after_dumping.yml",
            {"policies"},
        ),  # comments in various positions
        (
            "config_empty_en.yml",
            "config_empty_en_after_dumping.yml",
            {"policies", "pipeline"},
        ),  # no empty lines
        (
            "config_empty_fr.yml",
            "config_empty_fr_after_dumping.yml",
            {"policies", "pipeline"},
        ),  # no empty lines, with different language
        (
            "config_with_comments_after_dumping.yml",
            "config_with_comments_after_dumping.yml",
            {"policies"},
        ),  # with previous auto config that needs to be overwritten
    ],
)
def test_dump_config(
    tmp_path: Path,
    input_file: Text,
    expected_file: Text,
    capsys: CaptureFixture,
    autoconfig_keys: Set[Text],
):
    config_file = str(tmp_path / "config.yml")
    shutil.copyfile(str(CONFIG_FOLDER / input_file), config_file)

    autoconfig.get_configuration(config_file)

    actual = rasa.shared.utils.io.read_file(config_file)

    expected = rasa.shared.utils.io.read_file(str(CONFIG_FOLDER / expected_file))

    assert actual == expected

    captured = capsys.readouterr()
    assert "does not exist or is empty" not in captured.out

    for k in CONFIG_AUTOCONFIGURABLE_KEYS:
        if k in autoconfig_keys:
            assert k in captured.out
        else:
            assert k not in captured.out


@pytest.mark.parametrize(
    "input_file, expected_file, training_type",
    [
        (
            "config_empty_en.yml",
            "config_empty_en_after_dumping.yml",
            autoconfig.TrainingType.BOTH,
        ),
        (
            "config_empty_en.yml",
            "config_empty_en_after_dumping_core.yml",
            autoconfig.TrainingType.CORE,
        ),
        (
            "config_empty_en.yml",
            "config_empty_en_after_dumping_nlu.yml",
            autoconfig.TrainingType.NLU,
        ),
    ],
)
def test_get_configuration_for_different_training_types(
    tmp_path: Path,
    input_file: Text,
    expected_file: Text,
    training_type: autoconfig.TrainingType,
):
    config_file = str(tmp_path / "config.yml")
    shutil.copyfile(str(CONFIG_FOLDER / input_file), config_file)

    autoconfig.get_configuration(config_file, training_type)

    actual = rasa.shared.utils.io.read_file(config_file)

    expected = rasa.shared.utils.io.read_file(str(CONFIG_FOLDER / expected_file))

    assert actual == expected


def test_comment_causing_invalid_autoconfig(tmp_path: Path):
    """Regression test for https://github.com/RasaHQ/rasa/issues/6948."""

    config_file = tmp_path / "config.yml"
    shutil.copyfile(
        str(CONFIG_FOLDER / "config_with_comment_between_suggestions.yml"), config_file
    )

    _ = autoconfig.get_configuration(str(config_file))

    # This should not throw
    dumped = rasa.shared.utils.io.read_yaml_file(config_file)

    assert dumped
