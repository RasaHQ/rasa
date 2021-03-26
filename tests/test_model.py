import asyncio
import os
import tempfile
import time
import shutil
from pathlib import Path
from typing import Text, Optional, Any
from unittest import mock
from unittest.mock import Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch

import rasa
import rasa.constants
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_CORE_SUBDIRECTORY_NAME,
)
from rasa.shared.core.domain import KEY_RESPONSES
from rasa.shared.core.domain import Domain
import rasa.shared.utils.io
from rasa import model
from rasa.model import (
    FINGERPRINT_CONFIG_KEY,
    FINGERPRINT_CONFIG_WITHOUT_EPOCHS_KEY,
    FINGERPRINT_DOMAIN_WITHOUT_NLG_KEY,
    FINGERPRINT_NLG_KEY,
    FINGERPRINT_FILE_PATH,
    FINGERPRINT_NLU_DATA_KEY,
    FINGERPRINT_RASA_VERSION_KEY,
    FINGERPRINT_STORIES_KEY,
    FINGERPRINT_TRAINED_AT_KEY,
    FINGERPRINT_CONFIG_CORE_KEY,
    FINGERPRINT_CONFIG_NLU_KEY,
    SECTION_CORE,
    SECTION_NLU,
    can_finetune,
    create_package_rasa,
    get_latest_model,
    get_model,
    get_model_subdirectories,
    model_fingerprint,
    Fingerprint,
    did_section_fingerprint_change,
    should_retrain,
    FingerprintComparisonResult,
)
from rasa.exceptions import ModelNotFound
from rasa.model_training import train_core_async


def test_get_latest_model(trained_rasa_model: str):
    path_of_latest = os.path.join(os.path.dirname(trained_rasa_model), "latest.tar.gz")
    shutil.copy(trained_rasa_model, path_of_latest)

    model_directory = os.path.dirname(path_of_latest)

    assert get_latest_model(model_directory) == path_of_latest


def test_get_model_from_directory(trained_rasa_model: str):
    unpacked = get_model(trained_rasa_model)

    assert os.path.exists(os.path.join(unpacked, DEFAULT_CORE_SUBDIRECTORY_NAME))
    assert os.path.exists(os.path.join(unpacked, "nlu"))


def test_get_model_context_manager(trained_rasa_model: str):
    with get_model(trained_rasa_model) as unpacked:
        assert os.path.exists(unpacked)

    assert not os.path.exists(unpacked)


@pytest.mark.parametrize("model_path", ["foobar", "rasa", "README.md", None])
def test_get_model_exception(model_path: Optional[Text]):
    with pytest.raises(ModelNotFound):
        get_model(model_path)


def test_get_model_from_directory_with_subdirectories(
    trained_rasa_model: Text, tmp_path: Path
):
    unpacked = get_model(trained_rasa_model)
    unpacked_core, unpacked_nlu = get_model_subdirectories(unpacked)

    assert unpacked_core
    assert unpacked_nlu

    with pytest.raises(ModelNotFound):
        get_model_subdirectories(str(tmp_path))  # temp path should be empty


def test_get_model_from_directory_nlu_only(trained_rasa_model: Text):
    unpacked = get_model(trained_rasa_model)
    shutil.rmtree(os.path.join(unpacked, DEFAULT_CORE_SUBDIRECTORY_NAME))
    unpacked_core, unpacked_nlu = get_model_subdirectories(unpacked)

    assert not unpacked_core
    assert unpacked_nlu


def _fingerprint(
    config: Optional[Any] = None,
    config_nlu: Optional[Any] = None,
    config_core: Optional[Any] = None,
    config_without_epochs: Optional[Any] = None,
    domain: Optional[Any] = None,
    nlg: Optional[Any] = None,
    stories: Optional[Any] = None,
    nlu: Optional[Any] = None,
    rasa_version: Text = "1.0",
) -> Fingerprint:
    return {
        FINGERPRINT_CONFIG_KEY: config if config is not None else ["test"],
        FINGERPRINT_CONFIG_CORE_KEY: config_core
        if config_core is not None
        else ["test"],
        FINGERPRINT_CONFIG_NLU_KEY: config_nlu if config_nlu is not None else ["test"],
        FINGERPRINT_CONFIG_WITHOUT_EPOCHS_KEY: config_without_epochs
        if config_without_epochs
        else ["test"],
        FINGERPRINT_DOMAIN_WITHOUT_NLG_KEY: domain if domain is not None else ["test"],
        FINGERPRINT_NLG_KEY: nlg if nlg is not None else ["test"],
        FINGERPRINT_TRAINED_AT_KEY: time.time(),
        FINGERPRINT_RASA_VERSION_KEY: rasa_version,
        FINGERPRINT_STORIES_KEY: stories if stories is not None else ["test"],
        FINGERPRINT_NLU_DATA_KEY: nlu if nlu is not None else ["test"],
    }


def test_persist_and_load_fingerprint():
    from rasa.model import persist_fingerprint, fingerprint_from_path

    fingerprint = _fingerprint()
    output_directory = tempfile.mkdtemp()

    persist_fingerprint(output_directory, fingerprint)
    actual = fingerprint_from_path(output_directory)

    assert actual == fingerprint


@pytest.mark.parametrize(
    "fingerprint2, changed",
    [
        (_fingerprint(config=["other"]), True),
        (_fingerprint(config_core=["other"]), True),
        (_fingerprint(domain=["other"]), True),
        (_fingerprint(domain=Domain.empty()), True),
        (_fingerprint(stories=["test", "other"]), True),
        (_fingerprint(rasa_version="100"), True),
        (_fingerprint(config=["other"], domain=["other"]), True),
        (_fingerprint(nlg=["other"]), False),
        (_fingerprint(nlu=["test", "other"]), False),
        (_fingerprint(config_nlu=["other"]), False),
        (_fingerprint(config_without_epochs=["other"]), False),
    ],
)
def test_core_fingerprint_changed(fingerprint2: Fingerprint, changed: bool):
    fingerprint1 = _fingerprint()
    assert (
        did_section_fingerprint_change(fingerprint1, fingerprint2, SECTION_CORE)
        is changed
    )


@pytest.mark.parametrize(
    "fingerprint2, changed",
    [
        (_fingerprint(config=["other"]), True),
        (_fingerprint(nlu=["test", "other"]), True),
        (_fingerprint(rasa_version="100"), True),
        (_fingerprint(rasa_version="100", config=["other"]), True),
        (_fingerprint(nlg=["other"]), False),
        (_fingerprint(config_core=["other"]), False),
        (_fingerprint(stories=["other"]), False),
        (_fingerprint(config_without_epochs=["other"]), False),
    ],
)
def test_nlu_fingerprint_changed(fingerprint2: Fingerprint, changed: bool):
    fingerprint1 = _fingerprint()
    assert (
        did_section_fingerprint_change(fingerprint1, fingerprint2, SECTION_NLU)
        is changed
    )


def _project_files(
    project: Text,
    config_file: Text = DEFAULT_CONFIG_PATH,
    domain: Text = DEFAULT_DOMAIN_PATH,
    training_files: Text = DEFAULT_DATA_PATH,
) -> TrainingDataImporter:
    paths = {
        "config_file": config_file,
        "domain_path": domain,
        "training_data_paths": training_files,
    }
    paths = {
        k: v if v is None or Path(v).is_absolute() else os.path.join(project, v)
        for k, v in paths.items()
    }
    paths["training_data_paths"] = [paths["training_data_paths"]]

    return RasaFileImporter(**paths)


@pytest.mark.parametrize(
    "domain_path",
    [
        DEFAULT_DOMAIN_PATH,
        str((Path(".") / "data/test_domains/default_with_mapping.yml").absolute()),
    ],
)
async def test_create_fingerprint_from_paths(project: Text, domain_path: Text):
    project_files = _project_files(project, domain=domain_path)

    assert await model_fingerprint(project_files)


async def test_fingerprinting_changed_response_text(project: Text):
    importer = _project_files(project)

    old_fingerprint = await model_fingerprint(importer)
    old_domain = await importer.get_domain()

    # Change NLG content but keep actions the same
    domain_with_changed_nlg = old_domain.as_dict()
    domain_with_changed_nlg[KEY_RESPONSES]["utter_greet"].append({"text": "hi"})
    domain_with_changed_nlg = Domain.from_dict(domain_with_changed_nlg)

    importer.get_domain = asyncio.coroutine(lambda: domain_with_changed_nlg)

    new_fingerprint = await model_fingerprint(importer)

    assert (
        old_fingerprint[FINGERPRINT_DOMAIN_WITHOUT_NLG_KEY]
        == new_fingerprint[FINGERPRINT_DOMAIN_WITHOUT_NLG_KEY]
    )
    assert old_fingerprint[FINGERPRINT_NLG_KEY] != new_fingerprint[FINGERPRINT_NLG_KEY]


async def test_fingerprinting_changing_config_epochs(project: Text, tmp_path):
    config1 = {
        "language": "en",
        "pipeline": [
            {"name": "WhitespaceTokenizer"},
            {"name": "RegexFeaturizer"},
            {"name": "LexicalSyntacticFeaturizer"},
            {"name": "CountVectorsFeaturizer"},
            {
                "name": "CountVectorsFeaturizer",
                "analyzer": "char_wb",
                "min_ngram": 1,
                "max_ngram": 4,
            },
            {"name": "DIETClassifier", "epochs": 100},
            {"name": "EntitySynonymMapper"},
            {"name": "ResponseSelector", "epochs": 100},
            {
                "name": "FallbackClassifier",
                "threshold": 0.3,
                "ambiguity_threshold": 0.1,
            },
        ],
        "policies": [
            {"name": "MemoizationPolicy"},
            {"name": "TEDPolicy", "max_history": 5, "epochs": 100},
            {"name": "RulePolicy"},
        ],
    }

    config1_path = tmp_path / "config1.yml"
    rasa.shared.utils.io.write_yaml(config1, config1_path, True)
    importer = TrainingDataImporter.load_from_config(str(config1_path))
    old_fingerprint = await model_fingerprint(importer)

    config2 = {
        "language": "en",
        "pipeline": [
            {"name": "WhitespaceTokenizer"},
            {"name": "RegexFeaturizer"},
            {"name": "LexicalSyntacticFeaturizer"},
            {"name": "CountVectorsFeaturizer"},
            {
                "name": "CountVectorsFeaturizer",
                "analyzer": "char_wb",
                "min_ngram": 1,
                "max_ngram": 4,
            },
            {"name": "DIETClassifier", "epochs": 50},
            {"name": "EntitySynonymMapper"},
            {"name": "ResponseSelector", "epochs": 50},
            {
                "name": "FallbackClassifier",
                "threshold": 0.3,
                "ambiguity_threshold": 0.1,
            },
        ],
        "policies": [
            {"name": "MemoizationPolicy"},
            {"name": "TEDPolicy", "max_history": 5, "epochs": 50},
            {"name": "RulePolicy"},
        ],
    }

    config2_path = tmp_path / "config2.yml"
    rasa.shared.utils.io.write_yaml(config2, config2_path, True)
    importer = TrainingDataImporter.load_from_config(str(config2_path))
    new_fingerprint = await model_fingerprint(importer)

    assert (
        old_fingerprint[FINGERPRINT_CONFIG_WITHOUT_EPOCHS_KEY]
        == new_fingerprint[FINGERPRINT_CONFIG_WITHOUT_EPOCHS_KEY]
    )
    assert (
        old_fingerprint[FINGERPRINT_CONFIG_CORE_KEY]
        != new_fingerprint[FINGERPRINT_CONFIG_CORE_KEY]
    )
    assert (
        old_fingerprint[FINGERPRINT_CONFIG_NLU_KEY]
        != new_fingerprint[FINGERPRINT_CONFIG_NLU_KEY]
    )

    config3 = {
        "language": "en",
        "pipeline": [{"name": "WhitespaceTokenizer"},],
        "policies": [
            {"name": "MemoizationPolicy"},
            {"name": "TEDPolicy", "max_history": 5, "epochs": 50},
            {"name": "RulePolicy"},
        ],
    }

    config3_path = tmp_path / "config3.yml"
    rasa.shared.utils.io.write_yaml(config3, config3_path, True)
    importer = TrainingDataImporter.load_from_config(str(config3_path))
    new_fingerprint = await model_fingerprint(importer)

    assert (
        old_fingerprint[FINGERPRINT_CONFIG_WITHOUT_EPOCHS_KEY]
        != new_fingerprint[FINGERPRINT_CONFIG_WITHOUT_EPOCHS_KEY]
    )


@pytest.mark.parametrize("empty_key", ["pipeline", "policies"])
async def test_fingerprinting_config_epochs_empty_pipeline_or_policies(
    project: Text, tmp_path: Path, empty_key: Text,
):
    config = {
        "language": "en",
        "pipeline": [{"name": "WhitespaceTokenizer"},],
        "policies": [{"name": "MemoizationPolicy"},],
    }

    config[empty_key] = None

    model._get_fingerprint_of_config_without_epochs(config)


async def test_fingerprinting_additional_action(project: Text):
    importer = _project_files(project)

    old_fingerprint = await model_fingerprint(importer)
    old_domain = await importer.get_domain()

    domain_with_new_action = old_domain.as_dict()
    domain_with_new_action[KEY_RESPONSES]["utter_new"] = [{"text": "hi"}]
    domain_with_new_action = Domain.from_dict(domain_with_new_action)

    importer.get_domain = asyncio.coroutine(lambda: domain_with_new_action)

    new_fingerprint = await model_fingerprint(importer)

    assert (
        old_fingerprint[FINGERPRINT_DOMAIN_WITHOUT_NLG_KEY]
        != new_fingerprint[FINGERPRINT_DOMAIN_WITHOUT_NLG_KEY]
    )
    assert old_fingerprint[FINGERPRINT_NLG_KEY] != new_fingerprint[FINGERPRINT_NLG_KEY]


@pytest.mark.parametrize("use_fingerprint", [True, False])
async def test_rasa_packaging(
    trained_rasa_model: Text, project: Text, use_fingerprint: bool, tmp_path: Path
):
    unpacked_model_path = get_model(trained_rasa_model)

    os.remove(os.path.join(unpacked_model_path, FINGERPRINT_FILE_PATH))
    if use_fingerprint:
        fingerprint = await model_fingerprint(_project_files(project))
    else:
        fingerprint = None

    output_path = str(tmp_path / "test.tar.gz")

    create_package_rasa(unpacked_model_path, output_path, fingerprint)

    unpacked = get_model(output_path)

    assert (
        os.path.exists(os.path.join(unpacked, FINGERPRINT_FILE_PATH)) == use_fingerprint
    )
    assert os.path.exists(os.path.join(unpacked, DEFAULT_CORE_SUBDIRECTORY_NAME))
    assert os.path.exists(os.path.join(unpacked, "nlu"))

    assert not os.path.exists(unpacked_model_path)


@pytest.mark.parametrize(
    "fingerprint",
    [
        {
            "new": _fingerprint(),
            "old": _fingerprint(stories=["others"]),
            "retrain_core": True,
            "retrain_nlu": False,
            "retrain_nlg": True,
        },
        {
            "new": _fingerprint(nlu=["others"]),
            "old": _fingerprint(),
            "retrain_core": False,
            "retrain_nlu": True,
            "retrain_nlg": False,
        },
        {
            "new": _fingerprint(config="others"),
            "old": _fingerprint(),
            "retrain_core": True,
            "retrain_nlu": True,
            "retrain_nlg": True,
        },
        {
            "new": _fingerprint(config_core="others"),
            "old": _fingerprint(),
            "retrain_core": True,
            "retrain_nlu": False,
            "retrain_nlg": True,
        },
        {
            "new": _fingerprint(),
            "old": _fingerprint(config_nlu="others"),
            "retrain_core": False,
            "retrain_nlu": True,
            "retrain_nlg": False,
        },
        {
            "new": _fingerprint(),
            "old": _fingerprint(),
            "retrain_core": False,
            "retrain_nlu": False,
            "retrain_nlg": False,
        },
        {
            "new": _fingerprint(),
            "old": _fingerprint(nlg=["others"]),
            "retrain_core": False,
            "retrain_nlu": False,
            "retrain_nlg": True,
        },
    ],
)
def test_should_retrain(
    trained_rasa_model: Text, fingerprint: Fingerprint, tmp_path: Path
):
    old_model = set_fingerprint(trained_rasa_model, fingerprint["old"], tmp_path)

    retrain = should_retrain(fingerprint["new"], old_model, str(tmp_path))

    assert retrain.should_retrain_core() == fingerprint["retrain_core"]
    assert retrain.should_retrain_nlg() == fingerprint["retrain_nlg"]
    assert retrain.should_retrain_nlu() == fingerprint["retrain_nlu"]


async def test_should_not_retrain_core(
    domain_path: Text, tmp_path: Path, stack_config_path: Text
):
    # Don't use `stories_path` as checkpoints currently break fingerprinting
    story_file = tmp_path / "simple_story.yml"
    story_file.write_text(
        """
stories:
- story: test_story
  steps:
  - intent: greet
  - action: utter_greet
    """
    )
    trained_model = await train_core_async(
        domain_path, stack_config_path, str(story_file), str(tmp_path)
    )

    importer = TrainingDataImporter.load_from_config(
        stack_config_path, domain_path, training_data_paths=[str(story_file)]
    )

    new_fingerprint = await model.model_fingerprint(importer)

    result = model.should_retrain(new_fingerprint, trained_model, tmp_path)

    assert not result.should_retrain_core()


def set_fingerprint(
    trained_rasa_model: Text, fingerprint: Fingerprint, tmp_path: Path
) -> Text:
    unpacked_model_path = get_model(trained_rasa_model)

    os.remove(os.path.join(unpacked_model_path, FINGERPRINT_FILE_PATH))

    output_path = str(tmp_path / "test.tar.gz")

    create_package_rasa(unpacked_model_path, output_path, fingerprint)

    return output_path


@pytest.mark.parametrize(
    "comparison_result,retrain_all,retrain_core,retrain_nlg,retrain_nlu",
    [
        (FingerprintComparisonResult(force_training=True), True, True, True, True),
        (
            FingerprintComparisonResult(core=True, nlu=False, nlg=False),
            True,
            True,
            True,
            False,
        ),
        (
            FingerprintComparisonResult(core=False, nlu=True, nlg=False),
            True,
            False,
            False,
            True,
        ),
        (
            FingerprintComparisonResult(core=True, nlu=True, nlg=False),
            True,
            True,
            True,
            True,
        ),
    ],
)
def test_fingerprint_comparison_result(
    comparison_result: FingerprintComparisonResult,
    retrain_all: bool,
    retrain_core: bool,
    retrain_nlg: bool,
    retrain_nlu: bool,
):
    assert comparison_result.is_training_required() == retrain_all
    assert comparison_result.should_retrain_core() == retrain_core
    assert comparison_result.should_retrain_nlg() == retrain_nlg
    assert comparison_result.should_retrain_nlu() == retrain_nlu


async def test_update_with_new_domain(trained_rasa_model: Text, tmpdir: Path):
    _ = model.unpack_model(trained_rasa_model, tmpdir)

    new_domain = Domain.empty()

    mocked_importer = Mock()

    async def get_domain() -> Domain:
        return new_domain

    mocked_importer.get_domain = get_domain

    await model.update_model_with_new_domain(mocked_importer, tmpdir)

    actual = Domain.load(tmpdir / DEFAULT_CORE_SUBDIRECTORY_NAME / DEFAULT_DOMAIN_PATH)

    assert actual.is_empty()


async def test_update_with_new_domain_preserves_domain(
    tmpdir: Path, domain_with_categorical_slot_path
):
    domain = Domain.load(domain_with_categorical_slot_path)

    core_directory = tmpdir / DEFAULT_CORE_SUBDIRECTORY_NAME
    core_directory.mkdir()

    domain.persist(str(core_directory / DEFAULT_DOMAIN_PATH))
    domain.persist_specification(core_directory)

    mocked_importer = Mock()

    async def get_domain() -> Domain:
        return Domain.load(domain_with_categorical_slot_path)

    mocked_importer.get_domain = get_domain

    await model.update_model_with_new_domain(mocked_importer, tmpdir)

    new_persisted = Domain.load(core_directory / DEFAULT_DOMAIN_PATH)
    new_persisted.compare_with_specification(str(core_directory))


@pytest.mark.parametrize(
    "min_compatible_version, old_model_version, can_tune",
    [("2.1.0", "2.1.0", True), ("2.0.0", "2.1.0", True), ("2.1.0", "2.0.0", False),],
)
async def test_can_finetune_min_version(
    project: Text,
    monkeypatch: MonkeyPatch,
    old_model_version: Text,
    min_compatible_version: Text,
    can_tune: bool,
):
    importer = _project_files(project)

    monkeypatch.setattr(
        rasa.constants, "MINIMUM_COMPATIBLE_VERSION", min_compatible_version
    )
    monkeypatch.setattr(rasa, "__version__", old_model_version)
    old_fingerprint = await model_fingerprint(importer)
    new_fingerprint = await model_fingerprint(importer)

    with mock.patch("rasa.model.MINIMUM_COMPATIBLE_VERSION", min_compatible_version):
        assert can_finetune(old_fingerprint, new_fingerprint) == can_tune


@pytest.mark.parametrize("empty_key", ["pipeline", "policies"])
async def test_fingerprinting_config_epochs_empty_pipeline_or_policies(
    project: Text, tmp_path: Path, empty_key: Text,
):
    config = {
        "language": "en",
        "pipeline": [{"name": "WhitespaceTokenizer"},],
        "policies": [{"name": "MemoizationPolicy"},],
    }

    config[empty_key] = None

    model._get_fingerprint_of_config_without_epochs(config)
