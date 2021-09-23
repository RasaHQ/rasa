from pathlib import Path
from typing import List, Optional, Text, Type, Dict

import pytest

from rasa.nlu import registry
import rasa.nlu.train
import rasa.nlu.components
import rasa.shared.nlu.training_data.loading
from rasa.nlu.components import Component, ComponentBuilder
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.exceptions import InvalidConfigException
from rasa.nlu.model import Interpreter, Metadata, Trainer


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
        component_builder.load_component(component_meta, "", Metadata({}))
    assert "Cannot find class" in str(excinfo.value)


async def test_example_component(
    component_builder: ComponentBuilder, tmp_path: Path, nlu_as_json_path: Text
):
    _config = RasaNLUModelConfig(
        {"pipeline": [{"name": "tests.nlu.example_component.MyComponent"}]}
    )

    (trainer, trained, persisted_path) = await rasa.nlu.train.train(
        _config,
        data=nlu_as_json_path,
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
    tmp_path: Path, nlu_as_json_path: Text
):
    _config = RasaNLUModelConfig(
        # config with a component that does not have a `name` property
        {"pipeline": [{"parameter": 4}]}
    )

    with pytest.raises(InvalidConfigException):
        await rasa.nlu.train.train(
            _config, data=nlu_as_json_path, path=str(tmp_path),
        )


async def test_validate_component_keys_raises_warning_on_invalid_key(
    tmp_path: Path, nlu_as_json_path: Text
):
    _config = RasaNLUModelConfig(
        # config with a component that does not have a `confidence_threshold ` property
        {"pipeline": [{"name": "WhitespaceTokenizer", "confidence_threshold": 0.7}]}
    )

    with pytest.warns(UserWarning) as record:
        await rasa.nlu.train.train(
            _config, data=nlu_as_json_path, path=str(tmp_path),
        )

    assert "You have provided an invalid key" in record[0].message.args[0]


@pytest.mark.parametrize(
    "pipeline_template,should_warn",
    [
        (
            [
                {"name": "WhitespaceTokenizer"},
                {"name": "LexicalSyntacticFeaturizer"},
                {"name": "CRFEntityExtractor"},
                {"name": "DIETClassifier"},
            ],
            True,
        ),
        (
            [
                {"name": "WhitespaceTokenizer"},
                {"name": "LexicalSyntacticFeaturizer"},
                {"name": "DIETClassifier"},
            ],
            False,
        ),
    ],
)
def test_warn_of_competing_extractors(
    pipeline_template: List[Dict[Text, Text]], should_warn: bool
):
    config = RasaNLUModelConfig({"pipeline": pipeline_template})
    trainer = Trainer(config)

    if should_warn:
        with pytest.warns(UserWarning):
            rasa.nlu.components.warn_of_competing_extractors(trainer.pipeline)
    else:
        with pytest.warns(None) as records:
            rasa.nlu.components.warn_of_competing_extractors(trainer.pipeline)

        assert len(records) == 0


@pytest.mark.parametrize(
    "pipeline_template,data_path,should_warn",
    [
        (
            [
                {"name": "WhitespaceTokenizer"},
                {"name": "LexicalSyntacticFeaturizer"},
                {"name": "RegexEntityExtractor"},
                {"name": "DIETClassifier"},
            ],
            "data/test/overlapping_regex_entities.yml",
            True,
        ),
        (
            [
                {"name": "WhitespaceTokenizer"},
                {"name": "LexicalSyntacticFeaturizer"},
                {"name": "RegexEntityExtractor"},
            ],
            "data/test/overlapping_regex_entities.yml",
            False,
        ),
        (
            [
                {"name": "WhitespaceTokenizer"},
                {"name": "LexicalSyntacticFeaturizer"},
                {"name": "DIETClassifier"},
            ],
            "data/test/overlapping_regex_entities.yml",
            False,
        ),
        (
            [
                {"name": "WhitespaceTokenizer"},
                {"name": "LexicalSyntacticFeaturizer"},
                {"name": "RegexEntityExtractor"},
                {"name": "DIETClassifier"},
            ],
            "data/examples/rasa/demo-rasa.yml",
            False,
        ),
    ],
)
def test_warn_of_competition_with_regex_extractor(
    pipeline_template: List[Dict[Text, Text]], data_path: Text, should_warn: bool
):
    training_data = rasa.shared.nlu.training_data.loading.load_data(data_path)

    config = RasaNLUModelConfig({"pipeline": pipeline_template})
    trainer = Trainer(config)

    if should_warn:
        with pytest.warns(UserWarning):
            rasa.nlu.components.warn_of_competition_with_regex_extractor(
                trainer.pipeline, training_data
            )
    else:
        with pytest.warns(None) as records:
            rasa.nlu.components.warn_of_competition_with_regex_extractor(
                trainer.pipeline, training_data
            )

        assert len(records) == 0


OVERLAP_TESTS_CONFIG = RasaNLUModelConfig(
    {
        "pipeline": [
            {"name": "WhitespaceTokenizer"},
            {"name": "RegexEntityExtractor", "use_lookup_tables": False},
            {"name": "RegexEntityExtractor", "use_regexes": False},
        ]
    }
)

OVERLAP_TESTS_DATA = "data/test/overlapping_regex_entities.yml"


async def test_do_not_warn_for_non_overlapping_entities(tmp_path: Path):
    _, interpreter, _ = await rasa.nlu.train.train(
        OVERLAP_TESTS_CONFIG, data=OVERLAP_TESTS_DATA, path=str(tmp_path)
    )

    msg = "I am looking for some pasta"
    with pytest.warns(None, match="overlapping") as records:
        parsed_msg = interpreter.parse(msg)

    assert len(parsed_msg.get("entities", [])) == 1
    assert len(records) == 0


async def test_warn_for_overlapping_entities(tmp_path: Path):
    _, interpreter, _ = await rasa.nlu.train.train(
        OVERLAP_TESTS_CONFIG, data=OVERLAP_TESTS_DATA, path=str(tmp_path)
    )

    msg = "I am looking for some pizza"
    with pytest.warns(None, match="overlapping") as records:
        parsed_msg = interpreter.parse(msg)

    assert len(parsed_msg.get("entities", [])) == 2
    assert len(records) == 1
    for word in ["pizza", "meal", "zz-words", "RegexEntityExtractor"]:
        assert word in records[0].message.args[0]


async def test_warn_only_once_for_overlapping_entities(tmp_path: Path):
    _, interpreter, _ = await rasa.nlu.train.train(
        OVERLAP_TESTS_CONFIG, data=OVERLAP_TESTS_DATA, path=str(tmp_path)
    )

    msg = "I am looking for some pizza"
    with pytest.warns(None, match="overlapping") as records:
        parsed_msg = interpreter.parse(msg)

    assert len(parsed_msg.get("entities", [])) == 2
    assert len(records) == 1
    for word in ["pizza", "meal", "zz-words", "RegexEntityExtractor"]:
        assert word in records[0].message.args[0]

    # parse again but this time without warning again
    with pytest.warns(None, match="overlapping") as records:
        parsed_again_msg = interpreter.parse(msg)

    assert len(parsed_again_msg.get("entities", [])) == 2
    assert len(records) == 0
