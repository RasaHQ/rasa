from pluggy import PluginManager

from rasa.plugin import plugin_manager


def test_plugin_manager():
    manager = plugin_manager()
    assert isinstance(manager, PluginManager)

    manager_2 = plugin_manager()
    assert manager_2 == manager
