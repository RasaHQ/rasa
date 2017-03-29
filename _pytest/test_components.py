from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import pytest

from rasa_nlu import config
from rasa_nlu import registry
from rasa_nlu.components import Component, fill_args


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_component_init_without_args(component_class):
    component = component_class()
    assert issubclass(component.__class__, Component), \
        "Either component can't be initialized without args, or it doesn't subclass component but is registered as one."


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_no_components_with_same_name(component_class):
    names = [cls.name for cls in registry.component_classes]
    assert names.count(component_class.name) == 1, \
        "There is more than one component named {}".format(component_class.name)


@pytest.mark.parametrize("pipeline_template", registry.registered_pipeline_templates)
def test_all_components_in_mode_templates_exist(pipeline_template):
    components = registry.registered_pipeline_templates[pipeline_template]
    for component in components:
        assert component in registry.registered_components, "Model template contains unknown component."


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_all_arguments_can_be_satisfied_during_init(component_class):
    # All available context arguments that will ever be generated during init
    component = component_class()
    context_arguments = {}
    for clz in registry.component_classes:
        for ctx_arg in clz.context_provides.get("pipeline_init", []):
            context_arguments[ctx_arg] = None

    filled_args = fill_args(component.pipeline_init_args(), context_arguments, config.DEFAULT_CONFIG)
    assert len(filled_args) == len(component.pipeline_init_args())


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_all_arguments_can_be_satisfied_during_train(component_class):
    # All available context arguments that will ever be generated during train
    # it might still happen, that in a certain pipeline configuration arguments can not be satisfied!
    component = component_class()
    context_arguments = {"training_data": None}
    for clz in registry.component_classes:
        for ctx_arg in clz.context_provides.get("pipeline_init", []):
            context_arguments[ctx_arg] = None
        for ctx_arg in clz.context_provides.get("train", []):
            context_arguments[ctx_arg] = None

    filled_args = fill_args(component.train_args(), context_arguments, config.DEFAULT_CONFIG)
    assert len(filled_args) == len(component.train_args())


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_all_arguments_can_be_satisfied_during_parse(component_class):
    # All available context arguments that will ever be generated during parse
    component = component_class()
    context_arguments = {"text": None}
    for clz in registry.component_classes:
        for ctx_arg in clz.context_provides.get("pipeline_init", []):
            context_arguments[ctx_arg] = None
        for ctx_arg in clz.context_provides.get("process", []):
            context_arguments[ctx_arg] = None

    filled_args = fill_args(component.process_args(), context_arguments, config.DEFAULT_CONFIG)
    assert len(filled_args) == len(component.process_args())
