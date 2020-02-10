import tempfile
import os
import shutil
from typing import Text

import pytest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.tmpdir import TempdirFactory

import rasa.model

from rasa.train import train_core, train_nlu, train
from tests.core.test_model import _fingerprint


@pytest.mark.parametrize(
    "parameters",
    [
        {"model_name": "test-1234", "prefix": None},
        {"model_name": None, "prefix": "core-"},
        {"model_name": None, "prefix": None},
    ],
)
def test_package_model(trained_rasa_model, parameters):
    output_path = tempfile.mkdtemp()
    train_path = rasa.model.unpack_model(trained_rasa_model)

    model_path = rasa.model.package_model(
        _fingerprint(),
        output_path,
        train_path,
        parameters["model_name"],
        parameters["prefix"],
    )

    assert os.path.exists(model_path)

    file_name = os.path.basename(model_path)

    if parameters["model_name"]:
        assert parameters["model_name"] in file_name

    if parameters["prefix"]:
        assert parameters["prefix"] in file_name

    assert file_name.endswith(".tar.gz")


def count_temp_rasa_files(directory: Text) -> int:
    return len(
        [
            entry
            for entry in os.listdir(directory)
            if not any(
                [
                    # Ignore the following files/directories:
                    entry == "__pycache__",  # Python bytecode
                    entry.endswith(".py")  # Temp .py files created by TF
                    # Anything else is considered to be created by Rasa
                ]
            )
        ]
    )


def test_train_temp_files(
    tmp_path: Text,
    monkeypatch: MonkeyPatch,
    default_domain_path: Text,
    default_stories_file: Text,
    default_stack_config: Text,
    default_nlu_data: Text,
):
    monkeypatch.setattr(tempfile, "tempdir", tmp_path)
    output = "test_train_temp_files_models"

    train(
        default_domain_path,
        default_stack_config,
        [default_stories_file, default_nlu_data],
        output=output,
        force_training=True,
    )

    assert count_temp_rasa_files(tempfile.tempdir) == 0

    # After training the model, try to do it again. This shouldn't try to train
    # a new model because nothing has been changed. It also shouldn't create
    # any temp files.
    train(
        default_domain_path,
        default_stack_config,
        [default_stories_file, default_nlu_data],
        output=output,
    )

    assert count_temp_rasa_files(tempfile.tempdir) == 0


def test_train_core_temp_files(
    tmp_path: Text,
    monkeypatch: MonkeyPatch,
    default_domain_path: Text,
    default_stories_file: Text,
    default_stack_config: Text,
):
    monkeypatch.setattr(tempfile, "tempdir", tmp_path)

    train_core(
        default_domain_path,
        default_stack_config,
        default_stories_file,
        output="test_train_core_temp_files_models",
    )

    assert count_temp_rasa_files(tempfile.tempdir) == 0


def test_train_nlu_temp_files(
    tmp_path: Text,
    monkeypatch: MonkeyPatch,
    default_stack_config: Text,
    default_nlu_data: Text,
):
    monkeypatch.setattr(tempfile, "tempdir", tmp_path)

    train_nlu(
        default_stack_config,
        default_nlu_data,
        output="test_train_nlu_temp_files_models",
    )

    assert count_temp_rasa_files(tempfile.tempdir) == 0
