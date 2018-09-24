import mock
from rasa_nlu.project import Project
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


def test_pre_load_model():
    with mock.patch.object(Project, 'load_model', return_value=None) as mock_load_model:
        with mock.patch.object(data_router.DataRouter, '_pre_load_model', return_value=None) as mock_pre_load_model:
            dr = data_router.DataRouter()
            dr.project_store['project_test'] = Project()
            dr._pre_load('project_test', 'model_test', None)
    mock_pre_load_model.assert_called_once_with('project_test', 'model_test')

    with mock.patch.object(Project, 'load_model', return_value=None) as mock_load_model:
        with mock.patch.object(data_router.DataRouter, '_pre_load_model', return_value=None) as mock_pre_load_model:
            dr = data_router.DataRouter()
            dr.project_store['project_test'] = Project()
            dr._pre_load('project_test', 'model_test', '/whatever/absolute/path')
    mock_pre_load_model.assert_called_once_with('project_test', 'model_test')


def test_pre_load_project():
    with mock.patch.object(Project, 'load_model', return_value=None) as mock_load_model:
        with mock.patch.object(data_router.DataRouter, '_pre_load_project', return_value=None) as mock_pre_load_project:
            dr = data_router.DataRouter()
            dr.project_store['project_test'] = Project()
            dr._pre_load('project_test', None, None)
    mock_pre_load_project.assert_called_once_with('project_test')

    with mock.patch.object(Project, 'load_model', return_value=None) as mock_load_model:
        with mock.patch.object(data_router.DataRouter, '_pre_load_project', return_value=None) as mock_pre_load_project:
            dr = data_router.DataRouter()
            dr.project_store['project_test'] = Project()
            dr._pre_load('project_test', None, '/some/absolute/path')
    mock_pre_load_project.assert_called_once_with('project_test')
