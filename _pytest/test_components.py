import pytest

from rasa_nlu import registry
from rasa_nlu.components import Component


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_component_init_without_args(component_class):
    component = component_class()
    assert issubclass(component.__class__, Component), \
        "Either component can't be initialized without args, or it doesn't subclass component but is registered as one."


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_no_components_with_same_name(component_class):
    names = map(lambda cls: cls.name, registry.component_classes)
    assert names.count(component_class.name) == 1, \
        "There is more than one component named {}".format(component_class.name)


@pytest.mark.parametrize("model_template", registry.registered_model_templates)
def test_no_components_with_same_name(model_template):
    components = registry.registered_model_templates[model_template]
    for component in components:
        assert component in registry.registered_components, "Model template contains unknown component."
