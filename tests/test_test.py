from pathlib import Path
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

import rasa.model
import rasa.cli.utils


def monkeypatch_get_latest_model(tmp_path: Path, monkeypatch: MonkeyPatch):

    latest_model = tmp_path / "my_test_model.tar.gz"

    def mock_get_latest_model():
        return str(latest_model)

    monkeypatch.setattr(rasa.model, "get_latest_model", mock_get_latest_model)


def test_test_core_models_in_directory_input_default(
    capsys: CaptureFixture, tmp_path: Path, monkeypatch: MonkeyPatch
):
    from rasa.test import _test_core_models_in_directory_input

    monkeypatch_get_latest_model(tmp_path, monkeypatch)

    latest_model = Path(rasa.model.get_latest_model())
    latest_model.touch()

    # Input: default model file
    # => Should return containing directory
    modeldir = rasa.model.get_latest_model()
    p = Path(modeldir)
    new_modeldir = _test_core_models_in_directory_input(modeldir)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert new_modeldir == str(p.parent)


def test_test_core_models_in_directory_input_file(
    capsys: CaptureFixture, tmp_path: Path, monkeypatch: MonkeyPatch
):
    from rasa.test import _test_core_models_in_directory_input

    monkeypatch_get_latest_model(tmp_path, monkeypatch)

    other_model = tmp_path / "my_test_model1.tar.gz"
    assert str(other_model) != rasa.model.get_latest_model()
    other_model.touch()

    # Input: some file
    # => Should return containing directory and print a warning
    modeldir = str(other_model)
    p = Path(modeldir)
    new_modeldir = _test_core_models_in_directory_input(modeldir)
    captured = capsys.readouterr()
    rasa.cli.utils.print_warning(
        "You passed a file as '--model'. Will use the directory containing this file instead."
    )
    warning = capsys.readouterr()
    assert captured.out == warning.out
    assert new_modeldir == str(p.parent)


def test_test_core_models_in_directory_input_other(
    capsys: CaptureFixture, tmp_path: Path, monkeypatch: MonkeyPatch
):
    from rasa.test import _test_core_models_in_directory_input

    monkeypatch_get_latest_model(tmp_path, monkeypatch)

    # Input: anything that is not an existing file
    # => Should return input
    modeldir = "random_dir"
    assert not Path(modeldir).is_file()
    new_modeldir = _test_core_models_in_directory_input(modeldir)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert new_modeldir == modeldir
