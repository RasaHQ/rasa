import pytest

from rasa_nlu import persistor


def test_if_persistor_class_has_list_projects_method():
    with pytest.raises(NotImplementedError):
        persistor.Persistor().list_projects()
