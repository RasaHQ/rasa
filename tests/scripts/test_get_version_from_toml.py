from scripts.get_version_from_toml import get_rasa_version_from_pyproject
import os


def test_version_from_toml():
    pyproject_file_path = os.path.dirname(__file__) + "/test.toml"
    expected = "3.7.1rc1"
    version = get_rasa_version_from_pyproject(pyproject_file=pyproject_file_path)
    assert version == expected
