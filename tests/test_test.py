from pathlib import Path
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

import rasa.model
import rasa.cli.utils


def monkeypatch_get_latest_model(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    latest_model = tmp_path / "my_test_model.tar.gz"
    monkeypatch.setattr(rasa.model, "get_latest_model", lambda: str(latest_model))


def test_get_sanitized_model_directory_when_not_passing_model(
    capsys: CaptureFixture, tmp_path: Path, monkeypatch: MonkeyPatch
):
    from rasa.test import _get_sanitized_model_directory

    monkeypatch_get_latest_model(tmp_path, monkeypatch)

    # Create a fake model on disk so that `is_file` returns `True`
    latest_model = Path(rasa.model.get_latest_model())
    latest_model.touch()

    # Input: default model file
    # => Should return containing directory
    new_modeldir = _get_sanitized_model_directory(str(latest_model))
    captured = capsys.readouterr()
    assert not captured.out
    assert new_modeldir == str(latest_model.parent)


def test_get_sanitized_model_directory_when_passing_model_file_explicitly(
    capsys: CaptureFixture, tmp_path: Path, monkeypatch: MonkeyPatch
):
    from rasa.test import _get_sanitized_model_directory

    monkeypatch_get_latest_model(tmp_path, monkeypatch)

    other_model = tmp_path / "my_test_model1.tar.gz"
    assert str(other_model) != rasa.model.get_latest_model()
    other_model.touch()

    # Input: some file
    # => Should return containing directory and print a warning
    new_modeldir = _get_sanitized_model_directory(str(other_model))
    captured = capsys.readouterr()
    assert captured.out
    assert new_modeldir == str(other_model.parent)


def test_get_sanitized_model_directory_when_passing_other_input(
    capsys: CaptureFixture, tmp_path: Path, monkeypatch: MonkeyPatch
):
    from rasa.test import _get_sanitized_model_directory

    monkeypatch_get_latest_model(tmp_path, monkeypatch)

    # Input: anything that is not an existing file
    # => Should return input
    modeldir = "random_dir"
    assert not Path(modeldir).is_file()
    new_modeldir = _get_sanitized_model_directory(modeldir)
    captured = capsys.readouterr()
    assert not captured.out
    assert new_modeldir == modeldir
