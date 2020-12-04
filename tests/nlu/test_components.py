from pathlib import Path
from typing import List, Optional, Text, Type

import pytest

from rasa.nlu import registry, train
from rasa.nlu.components import Component, ComponentBuilder, find_unavailable_packages
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.exceptions import InvalidConfigException
from rasa.nlu.model import Interpreter, Metadata
from tests.nlu.conftest import DEFAULT_DATA_PATH


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_no_components_with_same_name(component_class: Type[Component]):
    """The name of the components need to be unique as they will
    be referenced by name when defining processing pipelines."""

    names = [cls.name for cls in registry.component_classes]
    assert (
        names.count(component_class.name) == 1
    ), f"There is more than one component named {component_class.name}"


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_all_required_components_can_be_satisfied(component_class: Type[Component]):
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


def test_builder_create_by_module_path(
    component_builder: ComponentBuilder, blank_config: RasaNLUModelConfig
):
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
    component_builder: ComponentBuilder,
    blank_config: RasaNLUModelConfig,
    test_input: Text,
    expected_output: Text,
    error: Exception,
):

    with pytest.raises(error):
        component_config = {"name": test_input}
        component_builder.create_component(component_config, blank_config)


def test_builder_load_unknown(component_builder: ComponentBuilder):
    with pytest.raises(Exception) as excinfo:
        component_meta = {"name": "my_made_up_componment"}
        component_builder.load_component(component_meta, "", Metadata({}, None))
    assert "Cannot find class" in str(excinfo.value)


async def test_example_component(component_builder: ComponentBuilder, tmp_path: Path):
    _config = RasaNLUModelConfig(
        {"pipeline": [{"name": "tests.nlu.example_component.MyComponent"}]}
    )

    (trainer, trained, persisted_path) = await train(
        _config,
        data=DEFAULT_DATA_PATH,
        path=str(tmp_path),
        component_builder=component_builder,
    )

    assert trainer.pipeline

    loaded = Interpreter.load(persisted_path, component_builder)

    assert loaded.parse("test") is not None


@pytest.mark.parametrize(
    "supported_language_list, not_supported_language_list, language, expected",
    [
        # in following comments: VAL stands for any valid setting
        # for language is `None`
        (None, None, None, True),
        # (None, None)
        (None, None, "en", True),
        # (VAL, None)
        (["en"], None, "en", True),
        (["en"], None, "zh", False),
        # (VAL, [])
        (["en"], [], "en", True),
        (["en"], [], "zh", False),
        # (None, VAL)
        (None, ["en"], "en", False),
        (None, ["en"], "zh", True),
        # ([], VAL)
        ([], ["en"], "en", False),
        ([], ["en"], "zh", True),
    ],
)
def test_can_handle_language_logically_correctness(
    supported_language_list: Optional[List[Text]],
    not_supported_language_list: Optional[List[Text]],
    language: Text,
    expected: bool,
):
    from rasa.nlu.components import Component

    SampleComponent = type(
        "SampleComponent",
        (Component,),
        {
            "supported_language_list": supported_language_list,
            "not_supported_language_list": not_supported_language_list,
        },
    )

    assert SampleComponent.can_handle_language(language) == expected


@pytest.mark.parametrize(
    "supported_language_list, not_supported_language_list, expected_exec_msg",
    [
        # in following comments: VAL stands for any valid setting
        # (None, [])
        (None, [], "Empty lists for both"),
        # ([], None)
        ([], None, "Empty lists for both"),
        # ([], [])
        ([], [], "Empty lists for both"),
        # (VAL, VAL)
        (["en"], ["zh"], "Only one of"),
    ],
)
def test_can_handle_language_guard_clause(
    supported_language_list: Optional[List[Text]],
    not_supported_language_list: Optional[List[Text]],
    expected_exec_msg: Text,
):
    from rasa.nlu.components import Component
    from rasa.shared.exceptions import RasaException

    SampleComponent = type(
        "SampleComponent",
        (Component,),
        {
            "supported_language_list": supported_language_list,
            "not_supported_language_list": not_supported_language_list,
        },
    )

    with pytest.raises(RasaException) as excinfo:
        SampleComponent.can_handle_language("random_string")
    assert expected_exec_msg in str(excinfo.value)


async def test_validate_requirements_raises_exception_on_component_without_name(
    tmp_path: Path,
):
    _config = RasaNLUModelConfig(
        # config with a component that does not have a `name` property
        {"pipeline": [{"parameter": 4}]}
    )

    with pytest.raises(InvalidConfigException):
        await train(
            _config, data=DEFAULT_DATA_PATH, path=str(tmp_path),
        )


async def test_validate_component_keys_raises_warning_on_invalid_key(tmp_path: Path,):
    _config = RasaNLUModelConfig(
        # config with a component that does not have a `confidence_threshold ` property
        {"pipeline": [{"name": "WhitespaceTokenizer", "confidence_threshold": 0.7}]}
    )

    with pytest.warns(UserWarning) as record:
        await train(
            _config, data=DEFAULT_DATA_PATH, path=str(tmp_path),
        )

    assert "You have provided an invalid key" in record[0].message.args[0]
