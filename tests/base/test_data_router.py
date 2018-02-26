import pytest
import mock

from rasa_nlu import persistor
from rasa_nlu import data_router


def test_list_projects_in_cloud_method():
    class UniqueValue(object):
        pass

    def mocked_get_persistor(*args, **kwargs):
        class MockedClass(object):
            def list_projects(self):
                return [UniqueValue()]

        return MockedClass()

    def mocked_data_router_init(self, *args, **kwargs):
        self.config = None

    with mock.patch.object(persistor, 'get_persistor', mocked_get_persistor):
        with mock.patch.object(data_router.DataRouter, "__init__", mocked_data_router_init):
            with mock.patch.object(data_router.DataRouter, "__del__", lambda: None):
                return_value = data_router.DataRouter(None, None)._list_projects_in_cloud()
    assert isinstance(return_value[0], UniqueValue)
