from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mock

from rasa_nlu import data_router
from rasa_nlu import persistor


def test_list_projects_in_cloud_method():
    class UniqueValue(object):
        pass

    def mocked_get_persistor(*args, **kwargs):
        class MockedClass(object):
            def list_projects(self):
                return [UniqueValue()]

            def list_models(self, project):
                return [UniqueValue()]

        return MockedClass()

    def mocked_data_router_init(self, *args, **kwargs):
        self.config = None

    with mock.patch.object(persistor, 'get_persistor',
                           mocked_get_persistor):
        return_value = data_router.DataRouter()._list_projects_in_cloud()
    assert isinstance(return_value[0], UniqueValue)
