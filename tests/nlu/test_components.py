import pytest

from typing import Tuple
from rasa.nlu import registry
from rasa.nlu.components import find_unavailable_packages
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from tests.nlu import utilities


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_no_components_with_same_name(component_class):
    """The name of the components need to be unique as they will
    be referenced by name when defining processing pipelines."""

    names = [cls.name for cls in registry.component_classes]
    assert (
        names.count(component_class.name) == 1
    ), f"There is more than one component named {component_class.name}"


@pytest.mark.parametrize("pipeline_template", registry.registered_pipeline_templates)
def test_all_components_in_model_templates_exist(pipeline_template):
    """We provide a couple of ready to use pipelines, this test ensures
    all components referenced by name in the
    pipeline definitions are available."""

    components = registry.registered_pipeline_templates[pipeline_template]
    for component in components:
        assert (
            component["name"] in registry.registered_components
        ), "Model template contains unknown component."


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_all_required_components_can_be_satisfied(component_class):
    """Checks that all required_components are present in the registry."""

    def _required_component_in_registry(component):
        for previous_component in registry.component_classes:
            if issubclass(previous_component, component):
                return True
        return False

    missing_components = []
    for required_component in component_class.required_components():
        if not _required_component_in_registry(required_component):
            missing_components.append(required_component.name)

    assert missing_components == [], (
        f"There is no required components {missing_components} "
        f"for '{component_class.name}'."
    )


def test_find_unavailable_packages():
    unavailable = find_unavailable_packages(
        ["my_made_up_package_name", "io", "foo_bar", "foo_bar"]
    )
    assert unavailable == {"my_made_up_package_name", "foo_bar"}


def test_builder_create_by_module_path(component_builder, blank_config):
    from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer

    path = "rasa.nlu.featurizers.sparse_featurizer.regex_featurizer.RegexFeaturizer"
    component_config = {"name": path}
    component = component_builder.create_component(component_config, blank_config)
    assert type(component) == RegexFeaturizer


@pytest.mark.parametrize(
    "test_input, expected_output, error",
    [
        ("my_made_up_component", "Cannot find class", Exception),
        (
            "rasa.nlu.featurizers.regex_featurizer.MadeUpClass",
            "Failed to find class",
            Exception,
        ),
        ("made.up.path.RegexFeaturizer", "No module named", ModuleNotFoundError),
    ],
)
def test_create_component_exception_messages(
    component_builder, blank_config, test_input, expected_output, error
):

    with pytest.raises(error):
        component_config = {"name": test_input}
        component_builder.create_component(component_config, blank_config)


def test_builder_load_unknown(component_builder):
    with pytest.raises(Exception) as excinfo:
        component_meta = {"name": "my_made_up_componment"}
        component_builder.load_component(component_meta, "", Metadata({}, None))
    assert "Cannot find class" in str(excinfo.value)


async def test_example_component(component_builder, tmpdir_factory):
    conf = RasaNLUModelConfig(
        {"pipeline": [{"name": "tests.nlu.example_component.MyComponent"}]}
    )

    interpreter = await utilities.interpreter_for(
        component_builder,
        data="./data/examples/rasa/demo-rasa.json",
        path=tmpdir_factory.mktemp("projects").strpath,
        config=conf,
    )

    r = interpreter.parse("test")
    assert r is not None
