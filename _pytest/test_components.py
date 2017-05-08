from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest

from rasa_nlu import registry
from rasa_nlu.components import fill_args
from rasa_nlu.extractors import EntityExtractor


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_no_components_with_same_name(component_class):
    """The name of the components need to be unique as they will be referenced by name
    when defining processing pipelines."""

    names = [cls.name for cls in registry.component_classes]
    assert names.count(component_class.name) == 1, \
        "There is more than one component named {}".format(component_class.name)


@pytest.mark.parametrize("pipeline_template", registry.registered_pipeline_templates)
def test_all_components_in_model_templates_exist(pipeline_template):
    """We provide a couple of ready to use pipelines, this test ensures all components referenced by name in the
    pipeline definitions are available."""

    components = registry.registered_pipeline_templates[pipeline_template]
    for component in components:
        assert component in registry.registered_components, "Model template contains unknown component."


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_all_arguments_can_be_satisfied_during_init(component_class, default_config, component_builder):
    """Check that `pipeline_init` method parameters can be filled filled from the context.

    The parameters declared on the `pipeline_init` are not filled directly, rather the method is called via reflection.
    During the reflection, the parameters are filled from a so called context that is created when creating the
    pipeline and gets initialized with the configuration values. To make sure all arguments `pipeline_init` declares
    can be provided during the reflection, we do a 'dry run' where we check all parameters are part of the context."""

    # All available context arguments that will ever be generated during init
    component = component_builder.create_component(component_class.name, default_config)
    context_arguments = {}
    for clz in registry.component_classes:
        for ctx_arg in clz.context_provides.get("pipeline_init", []):
            context_arguments[ctx_arg] = None

    filled_args = fill_args(component.pipeline_init_args(), context_arguments, default_config.as_dict())
    assert len(filled_args) == len(component.pipeline_init_args())


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_all_arguments_can_be_satisfied_during_train(component_class, default_config, component_builder):
    """Check that `train` method parameters can be filled filled from the context. Similar to `pipeline_init` test."""

    # All available context arguments that will ever be generated during train
    # it might still happen, that in a certain pipeline configuration arguments can not be satisfied!
    component = component_builder.create_component(component_class.name, default_config)
    context_arguments = {"training_data": None}
    for clz in registry.component_classes:
        for ctx_arg in clz.context_provides.get("pipeline_init", []):
            context_arguments[ctx_arg] = None
        for ctx_arg in clz.context_provides.get("train", []):
            context_arguments[ctx_arg] = None

    filled_args = fill_args(component.train_args(), context_arguments, default_config.as_dict())
    assert len(filled_args) == len(component.train_args())


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_all_arguments_can_be_satisfied_during_parse(component_class, default_config, component_builder):
    """Check that `parse` method parameters can be filled filled from the context. Similar to `pipeline_init` test."""

    # All available context arguments that will ever be generated during parse
    component = component_builder.create_component(component_class.name, default_config)
    context_arguments = {"text": None}
    for clz in registry.component_classes:
        for ctx_arg in clz.context_provides.get("pipeline_init", []):
            context_arguments[ctx_arg] = None
        for ctx_arg in clz.context_provides.get("process", []):
            context_arguments[ctx_arg] = None

    filled_args = fill_args(component.process_args(), context_arguments, default_config.as_dict())
    assert len(filled_args) == len(component.process_args())


def test_all_extractors_use_previous_entities():
    extractors = [c for c in registry.component_classes if isinstance(c, EntityExtractor)]
    assert all(["entities" in ex.process_args() for ex in extractors])
