import mock
import os
from moto import mock_s3

from rasa.nlu import persistor, train
from tests.nlu import utilities


class Object(object):
    pass


# noinspection PyPep8Naming
@mock_s3
def test_list_method_method_in_AWSPersistor(component_builder, tmpdir):
    # artificially create a persisted model
    _config = utilities.base_test_conf("keyword")
    os.environ["BUCKET_NAME"] = "rasa-test"
    os.environ["AWS_DEFAULT_REGION"] = "us-west-1"

    (trained, _, persisted_path) = train(
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
def test_list_models_method_raise_exeception_in_AWSPersistor():
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    awspersistor = persistor.AWSPersistor("rasa-test")
    result = awspersistor.list_models()

    assert result == []


# noinspection PyPep8Naming
def test_list_models_method_in_GCSPersistor():
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

    with mock.patch.object(persistor.GCSPersistor, "__init__", mocked_init):
        result = persistor.GCSPersistor("").list_models()

    assert result == ["model_name"]


# noinspection PyPep8Naming
def test_list_models_method_raise_exeception_in_GCSPersistor():
    # noinspection PyUnusedLocal
    def mocked_init(self, *args, **kwargs):
        self._model_dir_and_model_from_filename = lambda x: {
            "blob_name": ("project", "model_name")
        }[x]
        self.bucket = Object()

        def mocked_list_blobs():
            raise ValueError

        self.bucket.list_blobs = mocked_list_blobs

    with mock.patch.object(persistor.GCSPersistor, "__init__", mocked_init):
        result = persistor.GCSPersistor("").list_models()

    assert result == []


# noinspection PyPep8Naming
def test_list_models_method_in_AzurePersistor():
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

    with mock.patch.object(persistor.AzurePersistor, "__init__", mocked_init):
        result = persistor.AzurePersistor("", "", "").list_models()

    assert result == ["model_name"]


# noinspection PyPep8Naming
def test_list_models_method_raise_exeception_in_AzurePersistor():
    def mocked_init(self, *args, **kwargs):
        self._model_dir_and_model_from_filename = lambda x: {"blob_name": ("project",)}[
            x
        ]
        self.blob_client = Object()

        # noinspection PyUnusedLocal
        def mocked_list_blobs(container_name, prefix=None):
            raise ValueError

        self.blob_client.list_blobs = mocked_list_blobs

    with mock.patch.object(persistor.AzurePersistor, "__init__", mocked_init):
        result = persistor.AzurePersistor("", "", "").list_models()

    assert result == []
