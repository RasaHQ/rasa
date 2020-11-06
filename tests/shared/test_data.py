import glob
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import rasa.shared

import rasa.shared.data
from rasa.shared.constants import DEFAULT_E2E_TESTS_PATH
from rasa.shared.nlu.training_data.loading import load_data
from rasa.shared.utils.io import write_text_file, json_to_string
from tests.conftest import DEFAULT_NLU_DATA
from tests.core.conftest import DEFAULT_STORIES_FILE


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


def test_story_file_can_not_be_yml(tmpdir: Path):
    p = tmpdir / "test_non_md.yml"
    Path(p).touch()
    assert rasa.shared.data.is_story_file(str()) is False


def test_empty_story_file_is_not_story_file(tmpdir: Path):
    p = tmpdir / "test_non_md.md"
    Path(p).touch()
    assert rasa.shared.data.is_story_file(str(p)) is False


def test_story_file_with_minimal_story_is_story_file(tmpdir: Path):
    p = tmpdir / "story.md"
    s = """
## my story
    """
    write_text_file(s, p)
    assert rasa.shared.data.is_story_file(str(p))


def test_default_story_files_are_story_files():
    for fn in glob.glob(os.path.join("data", "test_stories", "*")):
        assert rasa.shared.data.is_story_file(fn)


def test_default_conversation_tests_are_conversation_tests_yml(tmpdir: Path):
    parent = tmpdir / DEFAULT_E2E_TESTS_PATH
    Path(parent).mkdir(parents=True)

    e2e_path = parent / "test_stories.yml"
    e2e_story = """stories:"""
    write_text_file(e2e_story, e2e_path)

    assert rasa.shared.data.is_test_stories_file(str(e2e_path))


def test_default_conversation_tests_are_conversation_tests_md(tmpdir: Path):
    # can be removed once conversation tests MD support is removed
    parent = tmpdir / DEFAULT_E2E_TESTS_PATH
    Path(parent).mkdir(parents=True)

    e2e_path = parent / "conversation_tests.md"
    e2e_story = """## my story test"""
    write_text_file(e2e_story, e2e_path)

    assert rasa.shared.data.is_test_stories_file(str(e2e_path))


def test_nlu_data_files_are_not_conversation_tests(tmpdir: Path):
    parent = tmpdir / DEFAULT_E2E_TESTS_PATH
    Path(parent).mkdir(parents=True)

    nlu_path = parent / "nlu.md"
    nlu_data = """
## intent: greet
- hello
- hi
- hallo
    """
    write_text_file(nlu_data, nlu_path)

    assert not rasa.shared.data.is_test_stories_file(str(nlu_path))


def test_domain_files_are_not_conversation_tests(tmpdir: Path):
    parent = tmpdir / DEFAULT_E2E_TESTS_PATH
    Path(parent).mkdir(parents=True)

    domain_path = parent / "domain.yml"

    assert not rasa.shared.data.is_test_stories_file(str(domain_path))


async def test_get_files_with_mixed_training_data():
    default_data_path = "data/test_mixed_yaml_training_data/training_data.yml"
    assert rasa.shared.data.get_data_files(
        default_data_path, rasa.shared.data.is_nlu_file
    )
    assert rasa.shared.data.get_data_files(
        default_data_path, rasa.shared.data.is_story_file
    )


def test_get_core_directory(project):
    data_dir = os.path.join(project, "data")
    core_directory = rasa.shared.data.get_core_directory([data_dir])
    core_files = os.listdir(core_directory)

    assert len(core_files) == 2
    assert any(file.endswith("stories.yml") for file in core_files)
    assert any(file.endswith("rules.yml") for file in core_files)


def test_get_nlu_directory(project):
    data_dir = os.path.join(project, "data")
    nlu_directory = rasa.shared.data.get_nlu_directory([data_dir])

    nlu_files = os.listdir(nlu_directory)

    assert len(nlu_files) == 1
    assert nlu_files[0].endswith("nlu.yml")


def test_get_nlu_file(project):
    data_file = os.path.join(project, "data/nlu.yml")
    nlu_directory = rasa.shared.data.get_nlu_directory(data_file)

    nlu_files = os.listdir(nlu_directory)

    original = load_data(data_file)
    copied = load_data(nlu_directory)

    assert nlu_files[0].endswith("nlu.yml")
    assert original.intent_examples == copied.intent_examples


def test_get_core_nlu_files(project):
    data_dir = os.path.join(project, "data")
    nlu_files = rasa.shared.data.get_data_files(
        [data_dir], rasa.shared.data.is_nlu_file
    )
    core_files = rasa.shared.data.get_data_files(
        [data_dir], rasa.shared.data.is_story_file
    )
    assert len(nlu_files) == 1
    assert list(nlu_files)[0].endswith("nlu.yml")

    assert len(core_files) == 2
    assert any(file.endswith("stories.yml") for file in core_files)
    assert any(file.endswith("rules.yml") for file in core_files)


def test_get_core_nlu_directories(project):
    data_dir = os.path.join(project, "data")
    core_directory, nlu_directory = rasa.shared.data.get_core_nlu_directories(
        [data_dir]
    )

    nlu_files = os.listdir(nlu_directory)

    assert len(nlu_files) == 1
    assert nlu_files[0].endswith("nlu.yml")

    core_files = os.listdir(core_directory)

    assert len(core_files) == 2
    assert any(file.endswith("stories.yml") for file in core_files)
    assert any(file.endswith("rules.yml") for file in core_files)


def test_get_core_nlu_directories_with_none():
    directories = rasa.shared.data.get_core_nlu_directories(None)

    assert all(directories)
    assert all(not os.listdir(directory) for directory in directories)


def test_same_file_names_get_resolved(tmp_path):
    # makes sure the resolution properly handles if there are two files with
    # with the same name in different directories

    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    data_dir_one = str(tmp_path / "one" / "stories.md")
    data_dir_two = str(tmp_path / "two" / "stories.md")
    shutil.copy2(DEFAULT_STORIES_FILE, data_dir_one)
    shutil.copy2(DEFAULT_STORIES_FILE, data_dir_two)

    nlu_dir_one = str(tmp_path / "one" / "nlu.yml")
    nlu_dir_two = str(tmp_path / "two" / "nlu.yml")
    shutil.copy2(DEFAULT_NLU_DATA, nlu_dir_one)
    shutil.copy2(DEFAULT_NLU_DATA, nlu_dir_two)

    core_directory, nlu_directory = rasa.shared.data.get_core_nlu_directories(
        [str(tmp_path)]
    )

    nlu_files = os.listdir(nlu_directory)

    assert len(nlu_files) == 2
    assert all(f.endswith("nlu.yml") for f in nlu_files)

    stories = os.listdir(core_directory)

    assert len(stories) == 2
    assert all(f.endswith("stories.md") for f in stories)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            "dialogflow",
            [
                "data/examples/dialogflow/agent.json",
                "data/examples/dialogflow/entities/cuisine.json",
                "data/examples/dialogflow/entities/cuisine_entries_en.json",
                "data/examples/dialogflow/entities/cuisine_entries_es.json",
                "data/examples/dialogflow/entities/flightNumber.json",
                "data/examples/dialogflow/entities/flightNumber_entries_en.json",
                "data/examples/dialogflow/entities/location.json",
                "data/examples/dialogflow/entities/location_entries_en.json",
                "data/examples/dialogflow/entities/location_entries_es.json",
                "data/examples/dialogflow/intents/Default Fallback Intent.json",
                "data/examples/dialogflow/intents/affirm.json",
                "data/examples/dialogflow/intents/affirm_usersays_en.json",
                "data/examples/dialogflow/intents/affirm_usersays_es.json",
                "data/examples/dialogflow/intents/goodbye.json",
                "data/examples/dialogflow/intents/goodbye_usersays_en.json",
                "data/examples/dialogflow/intents/goodbye_usersays_es.json",
                "data/examples/dialogflow/intents/hi.json",
                "data/examples/dialogflow/intents/hi_usersays_en.json",
                "data/examples/dialogflow/intents/hi_usersays_es.json",
                "data/examples/dialogflow/intents/inform.json",
                "data/examples/dialogflow/intents/inform_usersays_en.json",
                "data/examples/dialogflow/intents/inform_usersays_es.json",
                "data/examples/dialogflow/package.json",
            ],
        ),
        ("luis", ["data/examples/luis/demo-restaurants_v7.json",],),
        (
            "rasa",
            [
                "data/examples/rasa/demo-rasa-multi-intent.md",
                "data/examples/rasa/demo-rasa-responses.md",
                "data/examples/rasa/demo-rasa.json",
                "data/examples/rasa/demo-rasa.md",
            ],
        ),
        ("wit", ["data/examples/wit/demo-flights.json"]),
    ],
)
def test_find_nlu_files_with_different_formats(test_input, expected):
    examples_dir = "data/examples"
    data_dir = os.path.join(examples_dir, test_input)
    nlu_files = rasa.shared.data.get_data_files(
        [data_dir], rasa.shared.data.is_nlu_file
    )
    assert [Path(f) for f in nlu_files] == [Path(f) for f in expected]


def test_is_nlu_file_with_json():
    test = {
        "rasa_nlu_data": {
            "lookup_tables": [
                {"name": "plates", "elements": ["beans", "rice", "tacos", "cheese"]}
            ]
        }
    }

    directory = tempfile.mkdtemp()
    file = os.path.join(directory, "test.json")

    rasa.shared.utils.io.write_text_file(json_to_string(test), file)

    assert rasa.shared.data.is_nlu_file(file)


def test_is_not_nlu_file_with_json():
    directory = tempfile.mkdtemp()
    file = os.path.join(directory, "test.json")
    rasa.shared.utils.io.write_text_file('{"test": "a"}', file)

    assert not rasa.shared.data.is_nlu_file(file)


def test_get_story_file_with_yaml():
    examples_dir = "data/test_yaml_stories"
    core_files = rasa.shared.data.get_data_files(
        [examples_dir], rasa.shared.data.is_story_file
    )
    assert core_files
