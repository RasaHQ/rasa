from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mock
import pytest
from moto import mock_s3

from tests import utilities
from rasa_nlu import persistor


class Object(object):
    pass


def test_if_persistor_class_has_list_projects_method():
    with pytest.raises(NotImplementedError):
        persistor.Persistor().list_projects()


@mock_s3
def test_list_projects_method_in_AWSPersistor(component_builder):
    # artificially create a persisted model
    _config = utilities.base_test_conf("keyword")
    _config['storage'] = 'aws'
    _config['project'] = 'mytestproject'
    _config['aws_region'] = 'us-east-1'
    _config['bucket_name'] = 'rasa-test'
    (trained, persisted_path) = utilities.run_train(_config, component_builder)

    # We need to create the bucket since this is all in Moto's 'virtual' AWS
    # account
    awspersistor = persistor.AWSPersistor(_config['aws_region'],
                                          _config['bucket_name'])
    result = awspersistor.list_projects()

    assert result == ['mytestproject']


@mock_s3
def test_list_projects_method_raise_exeception_in_AWSPersistor():
    awspersistor = persistor.AWSPersistor("us-east-1", "rasa-test")
    result = awspersistor.list_projects()

    assert result == []


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


def test_list_projects_method_raise_exeception_in_GCSPersistor():
    def mocked_init(self, *args, **kwargs):
        self._project_and_model_from_filename = lambda x: {'blob_name': ('project', )}[x]
        self.bucket = Object()

        def mocked_list_blobs():
            raise ValueError

        self.bucket.list_blobs = mocked_list_blobs

    with mock.patch.object(persistor.GCSPersistor, "__init__", mocked_init):
        result = persistor.GCSPersistor("").list_projects()

    assert result == []
