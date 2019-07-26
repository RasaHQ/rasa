import tempfile
import os
import shutil

import pytest

from rasa.train import train

TEST_TEMP = "test_tmp"


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
    default_config_path,
    default_nlu_file,
):
    train(
        default_domain_path,
        default_config_path,
        [default_stories_file, default_nlu_file],
        force_training=True,
    )

    assert len(os.listdir(TEST_TEMP)) == 0

    # After training the model, try to do it again. This shouldn't try to train
    # a new model because nothing has been changed. It also shouldn't create
    # any temp files.
    train(
        default_domain_path,
        default_config_path,
        [default_stories_file, default_nlu_file],
    )

    assert len(os.listdir(TEST_TEMP)) == 0
