from typing import Text

import pytest
from unittest.mock import patch

from moto import mock_s3

from rasa.nlu import persistor
from rasa.nlu.persistor import Persistor


class Object:
    pass


# noinspection PyPep8Naming
def test_retrieve_tar_archive_with_s3_namespace():
    with mock_s3():
        model = "/my/s3/project/model.tar.gz"
        destination = "dst"
        with patch.object(persistor.AWSPersistor, "_copy") as copy:
            with patch.object(persistor.AWSPersistor, "_retrieve_tar") as retrieve:
                persistor.AWSPersistor("rasa-test", region_name="foo").retrieve(
                    model, destination
                )
            copy.assert_called_once_with("model.tar.gz", destination)
            retrieve.assert_called_once_with(model)


# noinspection PyPep8Naming
def test_s3_private_retrieve_tar():
    with mock_s3():
        # Ensure the S3 persistor writes to a filename `model.tar.gz`, whilst
        # passing the fully namespaced path to boto3
        model = "/my/s3/project/model.tar.gz"
        awsPersistor = persistor.AWSPersistor("rasa-test", region_name="foo")
        with patch.object(awsPersistor.bucket, "download_fileobj") as download_fileobj:
            # noinspection PyProtectedMember
            awsPersistor._retrieve_tar(model)
        retrieveArgs = download_fileobj.call_args[0]
        assert retrieveArgs[0] == model
        assert retrieveArgs[1].name == "model.tar.gz"


class TestPersistor(Persistor):
    def _retrieve_tar(self, filename: Text) -> Text:
        pass

    def _persist_tar(self, filekey: Text, tarname: Text) -> None:
        pass


def test_get_external_persistor():
    p = persistor.get_persistor("tests.nlu.test_persistor.TestPersistor")
    assert isinstance(p, TestPersistor)


def test_raise_exception_in_get_external_persistor():
    with pytest.raises(ImportError):
        _ = persistor.get_persistor("unknown.persistor")


# noinspection PyPep8Naming
@pytest.mark.parametrize(
    "model, archive", [("model.tar.gz", "model.tar.gz"), ("model", "model.tar.gz")]
)
def test_retrieve_tar_archive(model: Text, archive: Text):
    with patch.object(TestPersistor, "_copy") as f:
        with patch.object(TestPersistor, "_retrieve_tar") as f:
            TestPersistor().retrieve(model, "dst")
        f.assert_called_once_with(archive)
