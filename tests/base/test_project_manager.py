from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from mock import patch

from rasa_nlu.project_manager import ProjectManager


def test_list_projects_in_cloud_method():
    class UniqueValue(object):
        pass

    def mocked_get_persistor(*args, **kwargs):
        class MockedClass(object):
            def list_projects(self):
                return [UniqueValue()]

        return MockedClass()

    with patch("rasa_nlu.project_manager.get_persistor", mocked_get_persistor):
        return_value = ProjectManager('', '')._list_projects_in_cloud()  # noqa
    assert isinstance(return_value[0], UniqueValue)
