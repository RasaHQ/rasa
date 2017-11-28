import pytest
import mock

from rasa_nlu import persistor


class Object(object):
    pass


def test_if_persistor_class_has_list_projects_method():
    with pytest.raises(NotImplementedError):
        persistor.Persistor().list_projects()


def test_list_projects_method_in_AWSPersistor():
    def mocked_init(self, *args, **kwargs):
        self._project_and_model_from_filename = lambda x: {'project_key': ('project', 'model')}[x]
        self.bucket = Object()
        self.bucket.objects = Object()

        def mocked_filter():
            filter_result = Object()
            filter_result.key = 'project_key'
            return filter_result,

        self.bucket.objects.filter = mocked_filter

    with mock.patch.object(persistor.AWSPersistor, "__init__", mocked_init):
        result = persistor.AWSPersistor("", "", "").list_projects()

    assert result == ['project']


def test_list_projects_method_in_GCSPersistor():
    def mocked_init(self, *args, **kwargs):
        self._project_and_model_from_filename = lambda x: {'blob_name': ('project', )}[x]
        self.bucket = Object()

        def mocked_list_blobs():
            filter_result = Object()
            filter_result.name = 'blob_name'
            return filter_result,

        self.bucket.list_blobs = mocked_list_blobs

    with mock.patch.object(persistor.GCSPersistor, "__init__", mocked_init):
        result = persistor.GCSPersistor("").list_projects()

    assert result == ['project']
