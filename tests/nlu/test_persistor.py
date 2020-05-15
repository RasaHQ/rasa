import os
import pytest
from unittest.mock import patch

from moto import mock_s3

from rasa.nlu import persistor, train
from tests.nlu import utilities


class Object:
    pass


# noinspection PyPep8Naming
@mock_s3
async def test_list_method_method_in_AWS_persistor(component_builder, tmpdir):
    # artificially create a persisted model
    _config = utilities.base_test_conf("keyword")
    os.environ["BUCKET_NAME"] = "rasa-test"
    os.environ["AWS_DEFAULT_REGION"] = "us-west-1"

    (trained, _, persisted_path) = await train(
        _config,
        data="data/test/demo-rasa-small.json",
        path=tmpdir.strpath,
        storage="aws",
        component_builder=component_builder,
    )

    # We need to create the bucket since this is all in Moto's 'virtual' AWS
    # account
    awspersistor = persistor.AWSPersistor(os.environ["BUCKET_NAME"])
    result = awspersistor.list_models()

    assert len(result) == 1


# noinspection PyPep8Naming
@mock_s3
def test_list_models_method_raise_exeception_in_AWS_persistor():
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    awspersistor = persistor.AWSPersistor("rasa-test", region_name="foo")
    result = awspersistor.list_models()

    assert result == []


# noinspection PyPep8Naming
@mock_s3
def test_retrieve_tar_archive_with_s3_namespace():
    model = "/my/s3/project/model.tar.gz"
    destination = "dst"
    with patch.object(persistor.AWSPersistor, "_decompress") as decompress:
        with patch.object(persistor.AWSPersistor, "_retrieve_tar") as retrieve:
            persistor.AWSPersistor("rasa-test", region_name="foo").retrieve(
                model, destination
            )
        decompress.assert_called_once_with("model.tar.gz", destination)
        retrieve.assert_called_once_with(model)


# noinspection PyPep8Naming
@mock_s3
def test_s3_private_retrieve_tar():
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


# noinspection PyPep8Naming
def test_list_models_method_in_GCS_persistor():
    # noinspection PyUnusedLocal
    def mocked_init(self, *args, **kwargs):
        self._model_dir_and_model_from_filename = lambda x: {
            "blob_name": ("project", "model_name")
        }[x]
        self.bucket = Object()

        def mocked_list_blobs():
            filter_result = Object()
            filter_result.name = "blob_name"
            return (filter_result,)

        self.bucket.list_blobs = mocked_list_blobs

    with patch.object(persistor.GCSPersistor, "__init__", mocked_init):
        result = persistor.GCSPersistor("").list_models()

    assert result == ["model_name"]


# noinspection PyPep8Naming
def test_list_models_method_raise_exeception_in_GCS_persistor():
    # noinspection PyUnusedLocal
    def mocked_init(self, *args, **kwargs):
        self._model_dir_and_model_from_filename = lambda x: {
            "blob_name": ("project", "model_name")
        }[x]
        self.bucket = Object()

        def mocked_list_blobs():
            raise ValueError

        self.bucket.list_blobs = mocked_list_blobs

    with patch.object(persistor.GCSPersistor, "__init__", mocked_init):
        result = persistor.GCSPersistor("").list_models()

    assert result == []


# noinspection PyPep8Naming
def test_list_models_method_in_Azure_persistor():
    # noinspection PyUnusedLocal
    def mocked_init(self, *args, **kwargs):
        self._model_dir_and_model_from_filename = lambda x: {
            "blob_name": ("project", "model_name")
        }[x]
        self.blob_client = Object()
        self.container_name = "test"

        # noinspection PyUnusedLocal
        def mocked_list_blobs(container_name, prefix=None):
            filter_result = Object()
            filter_result.name = "blob_name"
            return (filter_result,)

        self.blob_client.list_blobs = mocked_list_blobs

    with patch.object(persistor.AzurePersistor, "__init__", mocked_init):
        result = persistor.AzurePersistor("", "", "").list_models()

    assert result == ["model_name"]


# noinspection PyPep8Naming
def test_list_models_method_raise_exeception_in_Azure_persistor():
    def mocked_init(self, *args, **kwargs):
        self._model_dir_and_model_from_filename = lambda x: {"blob_name": ("project",)}[
            x
        ]
        self.blob_client = Object()

        # noinspection PyUnusedLocal
        def mocked_list_blobs(container_name, prefix=None):
            raise ValueError

        self.blob_client.list_blobs = mocked_list_blobs

    with patch.object(persistor.AzurePersistor, "__init__", mocked_init):
        result = persistor.AzurePersistor("", "", "").list_models()

    assert result == []


# noinspection PyPep8Naming
@pytest.mark.parametrize(
    "model, archive", [("model.tar.gz", "model.tar.gz"), ("model", "model.tar.gz")]
)
def test_retrieve_tar_archive(model, archive):
    with patch.object(persistor.Persistor, "_decompress") as f:
        with patch.object(persistor.Persistor, "_retrieve_tar") as f:
            persistor.Persistor().retrieve(model, "dst")
        f.assert_called_once_with(archive)
