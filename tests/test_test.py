from pathlib import Path
from _pytest.monkeypatch import MonkeyPatch


def test_test_core_models_in_directory_default_input(
    capsys, tmp_path, monkeypatch: MonkeyPatch
):

    import rasa.model
    from rasa.test import _test_core_models_in_directory_default_input
    from rasa.cli.utils import print_warning

    latest_model = tmp_path / "my_test_model.tar.gz"
    latest_model.touch()
    other_model = tmp_path / "my_test_model1.tar.gz"
    other_model.touch()

    def mock_get_latest_model():
        return str(latest_model)

    monkeypatch.setattr(rasa.model, "get_latest_model", mock_get_latest_model)

    # Case 1: default model file given
    # => Should return containing directory
    modeldir = rasa.model.get_latest_model()
    p = Path(modeldir)
    new_modeldir = _test_core_models_in_directory_default_input(modeldir)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert new_modeldir == str(p.parent)

    # Case 2: another file given
    # => Should return containing directory and print a warning
    modeldir = str(other_model)
    new_modeldir = _test_core_models_in_directory_default_input(modeldir)
    captured = capsys.readouterr()
    print_warning(
        "You passed a file as '--model'. Will use the directory containing this file instead."
    )
    warning = capsys.readouterr()
    assert captured.out == warning.out
    assert new_modeldir == str(p.parent)

    # Case 3: anything else given (e.g. other existing directory or nonsense)
    # => Should return input
    modeldir = "random_dir"
    new_modeldir = _test_core_models_in_directory_default_input(modeldir)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert new_modeldir == modeldir
