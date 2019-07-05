import tempfile
import os
import shutil

import pytest

from rasa.model import unpack_model

from rasa.train import _package_model, train
from tests.core.test_model import _fingerprint

TEST_TEMP = "test_tmp"


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
    train_path = unpack_model(trained_rasa_model)

    model_path = _package_model(
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


@pytest.fixture
def move_tempdir():
    # Create a new *empty* tmp directory
    shutil.rmtree(TEST_TEMP, ignore_errors=True)
    os.mkdir(TEST_TEMP)
    tempfile.tempdir = TEST_TEMP
    yield
    tempfile.tempdir = None
    shutil.rmtree(TEST_TEMP)


def test_train_temp_files(
    move_tempdir,
    default_domain_path,
    default_stories_file,
    default_stack_config,
    default_nlu_data,
):
    train(
        default_domain_path,
        default_stack_config,
        [default_stories_file, default_nlu_data],
        force_training=True,
    )

    assert len(os.listdir(TEST_TEMP)) == 0

    # After training the model, try to do it again. This shouldn't try to train
    # a new model because nothing has been changed. It also shouldn't create
    # any temp files.
    train(
        default_domain_path,
        default_stack_config,
        [default_stories_file, default_nlu_data],
    )

    assert len(os.listdir(TEST_TEMP)) == 0
