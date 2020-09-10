import pytest

import rasa.shared.data


@pytest.mark.parametrize(
    "path,is_yaml",
    [
        ("my_file.yaml", True),
        ("my_file.yml", True),
        ("/a/b/c/my_file.yml", True),
        ("/a/b/c/my_file.ml", False),
        ("my_file.md", False),
    ],
)
def test_is_yaml_file(path, is_yaml):
    assert rasa.shared.data.is_likely_yaml_file(path) == is_yaml


@pytest.mark.parametrize(
    "path,is_md",
    [
        ("my_file.md", True),
        ("/a/b/c/my_file.md", True),
        ("/a/b/c/my_file.yml", False),
        ("my_file.yaml", False),
    ],
)
def test_is_md_file(path, is_md):
    assert rasa.shared.data.is_likely_markdown_file(path) == is_md


@pytest.mark.parametrize(
    "path,is_json",
    [
        ("my_file.json", True),
        ("/a/b/c/my_file.json", True),
        ("/a/b/c/my_file.yml", False),
        ("my_file.md", False),
    ],
)
def test_is_json_file(path, is_json):
    assert rasa.shared.data.is_likely_json_file(path) == is_json
