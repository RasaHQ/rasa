from typing import Text, Dict, Any, Set, List
import shutil

import pytest
from _pytest.capture import CaptureFixture
from pathlib import Path
from rasa.engine.constants import PLACEHOLDER_TRACKER
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.message import Message

import rasa.shared.utils.io
from rasa.shared.constants import ASSISTANT_ID_KEY, CONFIG_AUTOCONFIGURABLE_KEYS
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.engine.graph import GraphSchema, GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import (
    DefaultV1Recipe,
    DefaultV1RecipeRegisterException,
)
from rasa.engine.recipes.recipe import Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.validators.default_recipe_validator import (
    DefaultV1RecipeValidator,
)
from rasa.nlu.classifiers.mitie_intent_classifier import MitieIntentClassifier
from rasa.nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa.nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.data import TrainingType
import rasa.engine.validation
from rasa.shared.importers.rasa import RasaFileImporter


CONFIG_FOLDER = Path("data/test_config")

SOME_CONFIG = CONFIG_FOLDER / "stack_config.yml"
DEFAULT_CONFIG = Path("rasa/engine/recipes/config_files/default_config.yml")


def test_recipe_for_name():
    recipe = Recipe.recipe_for_name("default.v1")
    assert isinstance(recipe, DefaultV1Recipe)


@pytest.mark.parametrize(
    "config_path, expected_train_schema_path, expected_predict_schema_path, "
    "training_type, is_finetuning",
    [
        # The default config is the config which most users run
        (
            "rasa/engine/recipes/config_files/default_config.yml",
            "data/graph_schemas/default_config_e2e_train_schema.yml",
            "data/graph_schemas/default_config_e2e_predict_schema.yml",
            TrainingType.END_TO_END,
            False,
        ),
        # The default config without end to end
        (
            "rasa/engine/recipes/config_files/default_config.yml",
            "data/graph_schemas/default_config_train_schema.yml",
            "data/graph_schemas/default_config_predict_schema.yml",
            TrainingType.BOTH,
            False,
        ),
        (
            "rasa/engine/recipes/config_files/default_config.yml",
            "data/graph_schemas/default_config_core_train_schema.yml",
            "data/graph_schemas/default_config_core_predict_schema.yml",
            TrainingType.CORE,
            False,
        ),
        (
            "rasa/engine/recipes/config_files/default_config.yml",
            "data/graph_schemas/default_config_nlu_train_schema.yml",
            "data/graph_schemas/default_config_nlu_predict_schema.yml",
            TrainingType.NLU,
            False,
        ),
        # A config which uses Spacy and Duckling does not have Core model config
        (
            "data/test_config/config_pretrained_embeddings_spacy_duckling.yml",
            "data/graph_schemas/"
            "config_pretrained_embeddings_spacy_duckling_train_schema.yml",
            "data/graph_schemas/"
            "config_pretrained_embeddings_spacy_duckling_predict_schema.yml",
            TrainingType.BOTH,
            False,
        ),
        # A minimal NLU config without Core model
        (
            "data/test_config/keyword_classifier_config.yml",
            "data/graph_schemas/keyword_classifier_config_train_schema.yml",
            "data/graph_schemas/keyword_classifier_config_predict_schema.yml",
            TrainingType.BOTH,
            False,
        ),
        # A config which uses Mitie and does not have Core model
        (
            "data/test_config/config_pretrained_embeddings_mitie.yml",
            "data/graph_schemas/config_pretrained_embeddings_mitie_train_schema.yml",
            "data/graph_schemas/"
            "config_pretrained_embeddings_mitie_predict_schema.yml",
            TrainingType.BOTH,
            False,
        ),
        # A config which uses Mitie and Jiebe and does not have Core model
        (
            "data/test_config/config_pretrained_embeddings_mitie_zh.yml",
            "data/graph_schemas/config_pretrained_embeddings_mitie_zh_train_schema.yml",
            "data/graph_schemas/"
            "config_pretrained_embeddings_mitie_zh_predict_schema.yml",
            TrainingType.BOTH,
            False,
        ),
        # A core only model because of no pipeline
        (
            "data/test_config/max_hist_config.yml",
            "data/graph_schemas/max_hist_config_train_schema.yml",
            "data/graph_schemas/max_hist_config_predict_schema.yml",
            TrainingType.BOTH,
            False,
        ),
        # A full model which wants to be finetuned
        (
            "rasa/engine/recipes/config_files/default_config.yml",
            "data/graph_schemas/default_config_finetune_schema.yml",
            "data/graph_schemas/default_config_predict_schema.yml",
            TrainingType.BOTH,
            True,
        ),
    ],
)
def test_generate_graphs(
    config_path: Text,
    expected_train_schema_path: Text,
    expected_predict_schema_path: Text,
    training_type: TrainingType,
    is_finetuning: bool,
):
    expected_schema_as_dict = rasa.shared.utils.io.read_yaml_file(
        expected_train_schema_path
    )
    expected_train_schema = GraphSchema.from_dict(expected_schema_as_dict)

    expected_schema_as_dict = rasa.shared.utils.io.read_yaml_file(
        expected_predict_schema_path
    )
    expected_predict_schema = GraphSchema.from_dict(expected_schema_as_dict)

    config = rasa.shared.utils.io.read_yaml_file(config_path)

    recipe = Recipe.recipe_for_name(DefaultV1Recipe.name)
    model_config = recipe.graph_config_for_recipe(
        config, {}, training_type=training_type, is_finetuning=is_finetuning
    )

    train_schema = model_config.train_schema
    for node_name, node in expected_train_schema.nodes.items():
        assert train_schema.nodes[node_name] == node

    assert train_schema == expected_train_schema

    default_v1_validator = DefaultV1RecipeValidator(train_schema)
    importer = RasaFileImporter()
    # does not raise
    default_v1_validator.validate(importer)

    predict_schema = model_config.predict_schema
    for node_name, node in expected_predict_schema.nodes.items():
        assert predict_schema.nodes[node_name] == node

    assert predict_schema == expected_predict_schema

    rasa.engine.validation.validate(model_config)


@pytest.mark.parametrize(
    "cli_parameters, check_node, expected_config",
    [
        (
            {},
            "train_MitieIntentClassifier6",
            {"num_threads": 200000, "finetuning_epoch_fraction": 0.75},
        ),
        (
            {"num_threads": None},
            "train_MitieIntentClassifier6",
            {"num_threads": 200000, "finetuning_epoch_fraction": 0.75},
        ),
        (
            {"num_threads": 1},
            "train_MitieIntentClassifier6",
            {"num_threads": 1, "finetuning_epoch_fraction": 0.75},
        ),
        (
            {"num_threads": 1, "finetuning_epoch_fraction": 0.5},
            "train_MitieIntentClassifier6",
            # there is no `epochs` value specified so it doesn't get overridden
            {"num_threads": 1, "finetuning_epoch_fraction": 0.75},
        ),
        (
            {"finetuning_epoch_fraction": 0.5},
            "train_DIETClassifier7",
            {"epochs": 150, "num_threads": 200000, "finetuning_epoch_fraction": 0.5},
        ),
    ],
)
def test_nlu_config_doesnt_get_overridden(
    cli_parameters: Dict[Text, Any], check_node: Text, expected_config: Dict[Text, Any]
):
    config = rasa.shared.utils.io.read_yaml_file(
        "data/test_config/config_pretrained_embeddings_mitie_diet.yml"
    )
    recipe = Recipe.recipe_for_name(DefaultV1Recipe.name)
    model_config = recipe.graph_config_for_recipe(
        config, cli_parameters, training_type=TrainingType.BOTH, is_finetuning=True
    )

    train_schema = model_config.train_schema
    mitie_node = train_schema.nodes.get(check_node)
    assert mitie_node.config == expected_config


def test_language_returning():
    config = rasa.shared.utils.io.read_yaml(
        """
    language: "xy"
    version: '2.0'

    policies:
    - name: RulePolicy
    """
    )

    recipe = Recipe.recipe_for_name(DefaultV1Recipe.name)
    model_config = recipe.graph_config_for_recipe(config, {})

    assert model_config.language == "xy"


def test_tracker_generator_parameter_interpolation():
    config = rasa.shared.utils.io.read_yaml(
        """
    version: '2.0'

    policies:
    - name: RulePolicy
    """
    )

    augmentation = 0
    debug_plots = True

    recipe = Recipe.recipe_for_name(DefaultV1Recipe.name)
    model_config = recipe.graph_config_for_recipe(
        config, {"augmentation_factor": augmentation, "debug_plots": debug_plots}
    )

    node = model_config.train_schema.nodes["training_tracker_provider"]

    assert node.config == {
        "augmentation_factor": augmentation,
        "debug_plots": debug_plots,
    }


def test_nlu_training_data_persistence():
    config = rasa.shared.utils.io.read_yaml(
        """
    version: '2.0'

    pipeline:
    - name: KeywordIntentClassifier
    """
    )

    recipe = Recipe.recipe_for_name(DefaultV1Recipe.name)
    model_config = recipe.graph_config_for_recipe(
        config, {"persist_nlu_training_data": True}
    )

    node = model_config.train_schema.nodes["nlu_training_data_provider"]

    assert node.config == {"language": None, "persist": True}
    assert node.is_target


def test_num_threads_interpolation():
    expected_schema_as_dict = rasa.shared.utils.io.read_yaml_file(
        "data/graph_schemas/config_pretrained_embeddings_mitie_train_schema.yml"
    )
    expected_train_schema = GraphSchema.from_dict(expected_schema_as_dict)

    expected_schema_as_dict = rasa.shared.utils.io.read_yaml_file(
        "data/graph_schemas/config_pretrained_embeddings_mitie_predict_schema.yml"
    )
    expected_predict_schema = GraphSchema.from_dict(expected_schema_as_dict)

    for node_name, node in expected_train_schema.nodes.items():
        if issubclass(
            node.uses,
            (SklearnIntentClassifier, MitieEntityExtractor, MitieIntentClassifier),
        ) and node_name.startswith("train_"):
            node.config["num_threads"] = 20

    config = rasa.shared.utils.io.read_yaml_file(
        "data/test_config/config_pretrained_embeddings_mitie.yml"
    )

    recipe = Recipe.recipe_for_name(DefaultV1Recipe.name)
    model_config = recipe.graph_config_for_recipe(config, {"num_threads": 20})

    train_schema = model_config.train_schema
    for node_name, node in expected_train_schema.nodes.items():
        assert train_schema.nodes[node_name] == node

    assert train_schema == expected_train_schema

    predict_schema = model_config.predict_schema
    for node_name, node in expected_predict_schema.nodes.items():
        assert predict_schema.nodes[node_name] == node

    assert predict_schema == expected_predict_schema


def test_epoch_fraction_cli_param():
    expected_schema_as_dict = rasa.shared.utils.io.read_yaml_file(
        "data/graph_schemas/default_config_finetune_epoch_fraction_schema.yml"
    )
    expected_train_schema = GraphSchema.from_dict(expected_schema_as_dict)

    config = rasa.shared.utils.io.read_yaml_file(
        "rasa/engine/recipes/config_files/default_config.yml"
    )

    recipe = Recipe.recipe_for_name(DefaultV1Recipe.name)
    model_config = recipe.graph_config_for_recipe(
        config, {"finetuning_epoch_fraction": 0.5}, is_finetuning=True
    )

    train_schema = model_config.train_schema
    for node_name, node in expected_train_schema.nodes.items():
        assert train_schema.nodes[node_name] == node

    assert train_schema == expected_train_schema


def test_epoch_fraction_cli_param_unspecified():
    # TODO: enhance testing of cli instead of imitating expected parsed input
    expected_schema_as_dict = rasa.shared.utils.io.read_yaml_file(
        "data/graph_schemas/default_config_finetune_epoch_fraction_schema.yml"
    )
    expected_train_schema = GraphSchema.from_dict(expected_schema_as_dict)

    # modify the expected schema
    for schema_node in expected_train_schema.nodes.values():
        if "finetuning_epoch_fraction" in schema_node.config:
            schema_node.config["finetuning_epoch_fraction"] = 1.0
            if "epochs" in schema_node.config:
                schema_node.config["epochs"] *= 2

    config = rasa.shared.utils.io.read_yaml_file(
        "rasa/engine/recipes/config_files/default_config.yml"
    )

    recipe = Recipe.recipe_for_name(DefaultV1Recipe.name)
    model_config = recipe.graph_config_for_recipe(
        config, {"finetuning_epoch_fraction": None}, is_finetuning=True
    )

    train_schema = model_config.train_schema
    for node_name, node in expected_train_schema.nodes.items():
        assert train_schema.nodes[node_name] == node

    assert train_schema == expected_train_schema


def test_register_component():
    @DefaultV1Recipe.register(
        DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER,
        is_trainable=True,
        model_from="Herman",
    )
    class MyClassGraphComponent(GraphComponent):
        @classmethod
        def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
        ) -> GraphComponent:
            return cls()

    assert DefaultV1Recipe._from_registry(
        MyClassGraphComponent.__name__
    ) == DefaultV1Recipe.RegisteredComponent(
        MyClassGraphComponent,
        {DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER},
        True,
        "Herman",
    )
    assert MyClassGraphComponent()


def test_register_component_using_tracker():
    @DefaultV1Recipe.register(
        DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER,
        is_trainable=True,
        model_from="Herman",
    )
    class MyClassGraphComponent(GraphComponent):
        def process(
            self, messages: List[Message], tracker: DialogueStateTracker
        ) -> List[Message]:
            ...

    config = rasa.shared.utils.io.read_yaml(
        """
        language: "xy"
        version: '2.0'
        pipeline:
        - name: MyClassGraphComponent
        """
    )

    recipe = Recipe.recipe_for_name(DefaultV1Recipe.name)
    model_config = recipe.graph_config_for_recipe(config, {})

    node_in_graph = model_config.predict_schema.nodes.get("run_MyClassGraphComponent0")
    assert node_in_graph is not None
    # check that the node was configured to require the tracker as an input
    assert node_in_graph.needs.get("tracker") == PLACEHOLDER_TRACKER


def test_register_component_with_multiple_types():
    @DefaultV1Recipe.register(
        [
            DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER,
            DefaultV1Recipe.ComponentType.MODEL_LOADER,
        ],
        is_trainable=True,
        model_from="Herman",
    )
    class MyClassGraphComponent(GraphComponent):
        @classmethod
        def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
        ) -> GraphComponent:
            return cls()

    assert DefaultV1Recipe._from_registry(
        MyClassGraphComponent.__name__
    ) == DefaultV1Recipe.RegisteredComponent(
        MyClassGraphComponent,
        {
            DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER,
            DefaultV1Recipe.ComponentType.MODEL_LOADER,
        },
        True,
        "Herman",
    )
    assert MyClassGraphComponent()


def test_register_invalid_component():
    with pytest.raises(DefaultV1RecipeRegisterException):

        @DefaultV1Recipe.register(
            DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, False, "Bla"
        )
        class MyClass:
            pass


def test_retrieve_not_registered_class():
    class NotRegisteredClass:
        pass

    with pytest.raises(InvalidConfigException):
        # noinspection PyTypeChecker
        DefaultV1Recipe._from_registry(NotRegisteredClass.__name__)


def test_retrieve_via_module_path():
    model_config = DefaultV1Recipe().graph_config_for_recipe(
        {"policies": [{"name": "rasa.core.policies.ted_policy.TEDPolicy"}]},
        {},
        TrainingType.CORE,
    )

    assert any(
        issubclass(node.uses, TEDPolicy)
        for node in model_config.train_schema.nodes.values()
    )
    assert any(
        issubclass(node.uses, TEDPolicy)
        for node in model_config.predict_schema.nodes.values()
    )


def test_retrieve_via_invalid_module_path():
    with pytest.raises(ImportError):
        path = "rasa.core.policies.ted_policy.TEDPolicy1000"
        DefaultV1Recipe().graph_config_for_recipe(
            {"policies": [{"name": path}]}, {}, TrainingType.CORE
        )


def test_train_nlu_without_nlu_pipeline():
    with pytest.raises(InvalidConfigException):
        DefaultV1Recipe().graph_config_for_recipe(
            {"pipeline": []}, {}, TrainingType.NLU
        )


def test_train_core_without_nlu_pipeline():
    with pytest.raises(InvalidConfigException):
        DefaultV1Recipe().graph_config_for_recipe(
            {"policies": []}, {}, TrainingType.CORE
        )


@pytest.mark.parametrize(
    "config_path, expected_keys_to_configure",
    [
        (Path("rasa/cli/initial_project/config.yml"), {"pipeline", "policies"}),
        (CONFIG_FOLDER / "config_policies_empty.yml", {"policies"}),
        (CONFIG_FOLDER / "config_pipeline_empty.yml", {"pipeline"}),
        (CONFIG_FOLDER / "config_policies_missing.yml", {"policies"}),
        (CONFIG_FOLDER / "config_pipeline_missing.yml", {"pipeline"}),
        (SOME_CONFIG, set()),
    ],
)
def test_get_configuration(
    config_path: Path, expected_keys_to_configure: Set[Text], tmp_path: Path
):
    new_config_file = tmp_path / "new_config.yml"
    shutil.copyfile(config_path, new_config_file)

    config = rasa.shared.utils.io.read_model_configuration(new_config_file)
    _config, _missing_keys, configured_keys = DefaultV1Recipe.auto_configure(
        new_config_file, config
    )

    assert sorted(configured_keys) == sorted(expected_keys_to_configure)


@pytest.mark.parametrize(
    "language, keys_to_configure",
    [
        ("en", {"policies"}),
        ("en", {"pipeline"}),
        ("fr", {"pipeline"}),
        ("en", {"policies", "pipeline"}),
    ],
)
def test_auto_configure(language: Text, keys_to_configure: Set[Text]):
    expected_config = rasa.shared.utils.io.read_config_file(DEFAULT_CONFIG)

    config = DefaultV1Recipe.complete_config({"language": language}, keys_to_configure)

    for k in keys_to_configure:
        assert config[k] == expected_config[k]  # given keys are configured correctly

    assert config.get("language") == language
    config.pop("language")
    assert len(config) == len(keys_to_configure)  # no other keys are configured


@pytest.mark.parametrize(
    "config_path, missing_keys",
    [
        (CONFIG_FOLDER / "config_language_only.yml", {"pipeline", "policies"}),
        (CONFIG_FOLDER / "config_policies_missing.yml", {"policies"}),
        (CONFIG_FOLDER / "config_pipeline_missing.yml", {"pipeline"}),
        (SOME_CONFIG, []),
    ],
)
def test_add_missing_config_keys_to_file(
    tmp_path: Path, config_path: Path, missing_keys: Set[Text]
):
    config_file = str(tmp_path / "config.yml")
    shutil.copyfile(str(config_path), config_file)

    DefaultV1Recipe._add_missing_config_keys_to_file(config_file, missing_keys)

    config_after_addition = rasa.shared.utils.io.read_config_file(config_file)

    assert all(key in config_after_addition for key in missing_keys)


def test_dump_config_missing_file(tmp_path: Path, capsys: CaptureFixture):

    config_path = tmp_path / "non_existent_config.yml"

    config = rasa.shared.utils.io.read_config_file(str(SOME_CONFIG))

    DefaultV1Recipe._dump_config(config, str(config_path), set(), {"policies"})

    assert not config_path.exists()

    captured = capsys.readouterr()
    assert "has been removed or modified" in captured.out


# Test a few cases that are known to be potentially tricky (have failed in the past)
@pytest.mark.parametrize(
    "input_file, expected_file, autoconfig_keys",
    [
        (
            "config_with_comments.yml",
            "config_with_comments_after_dumping.yml",
            {"policies"},
        ),  # comments in various positions
        (
            "config_empty_en.yml",
            "config_empty_en_after_dumping.yml",
            {"policies", "pipeline"},
        ),  # no empty lines
        (
            "config_empty_fr.yml",
            "config_empty_fr_after_dumping.yml",
            {"policies", "pipeline"},
        ),  # no empty lines, with different language
        (
            "config_with_comments_after_dumping.yml",
            "config_with_comments_after_dumping.yml",
            {"policies"},
        ),  # with previous auto config that needs to be overwritten
    ],
)
def test_dump_config(
    tmp_path: Path,
    input_file: Text,
    expected_file: Text,
    capsys: CaptureFixture,
    autoconfig_keys: Set[Text],
):
    config_file = str(tmp_path / "config.yml")
    shutil.copyfile(str(CONFIG_FOLDER / input_file), config_file)
    old_config = rasa.shared.utils.io.read_model_configuration(config_file)
    DefaultV1Recipe.auto_configure(config_file, old_config)
    new_config = rasa.shared.utils.io.read_model_configuration(config_file)

    expected = rasa.shared.utils.io.read_model_configuration(
        CONFIG_FOLDER / expected_file
    )

    assert new_config == expected

    captured = capsys.readouterr()
    assert "does not exist or is empty" not in captured.out

    for k in CONFIG_AUTOCONFIGURABLE_KEYS:
        if k in autoconfig_keys:
            assert k in captured.out
        else:
            assert k not in captured.out


@pytest.mark.parametrize(
    "input_file, expected_file, training_type",
    [
        (
            "config_empty_en.yml",
            "config_empty_en_after_dumping.yml",
            TrainingType.BOTH,
        ),
        (
            "config_empty_en.yml",
            "config_empty_en_after_dumping_core.yml",
            TrainingType.CORE,
        ),
        (
            "config_empty_en.yml",
            "config_empty_en_after_dumping_nlu.yml",
            TrainingType.NLU,
        ),
    ],
)
def test_get_configuration_for_different_training_types(
    tmp_path: Path,
    input_file: Text,
    expected_file: Text,
    training_type: TrainingType,
):
    config_file = str(tmp_path / "config.yml")
    shutil.copyfile(str(CONFIG_FOLDER / input_file), config_file)
    config = rasa.shared.utils.io.read_model_configuration(config_file)

    DefaultV1Recipe.auto_configure(config_file, config, training_type)

    actual = rasa.shared.utils.io.read_file(config_file)

    expected = rasa.shared.utils.io.read_file(str(CONFIG_FOLDER / expected_file))

    assert actual == expected


def test_comment_causing_invalid_autoconfig(tmp_path: Path):
    """Regression test for https://github.com/RasaHQ/rasa/issues/6948."""
    config_file = tmp_path / "config.yml"
    shutil.copyfile(
        str(CONFIG_FOLDER / "config_with_comment_between_suggestions.yml"), config_file
    )
    config = rasa.shared.utils.io.read_model_configuration(config_file)

    _ = DefaultV1Recipe.auto_configure(str(config_file), config)

    # This should not throw
    dumped = rasa.shared.utils.io.read_yaml_file(config_file)

    assert dumped


def test_needs_from_args():
    @DefaultV1Recipe.register(
        DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER,
        is_trainable=True,
        model_from="Herman",
    )
    class MyClassGraphComponent(GraphComponent):
        @classmethod
        def run(
            cls,
            bar: Any,
            resource: Resource,
            foo: Any,
            training_trackers: Any,
            training_data: Any,
            tracker: Any,
        ) -> int:
            return 42

    assert DefaultV1Recipe()._get_needs_from_args(MyClassGraphComponent, "run") == {
        "bar": "bar_provider",
        "foo": "foo_provider",
        "resource": "resource_provider",
        "training_trackers": "training_tracker_provider",
        "training_data": "nlu_training_data_provider",
        "tracker": PLACEHOLDER_TRACKER,
    }


@pytest.mark.parametrize(
    "config_file",
    [
        "data/test_config/config_unique_assistant_id.yml",
        "data/test_config/config_defaults.yml",
    ],
)
def test_graph_config_for_recipe_with_assistant_id(config_file):
    config = rasa.shared.utils.io.read_model_configuration(config_file)

    recipe = Recipe.recipe_for_name(DefaultV1Recipe.name)
    model_config = recipe.graph_config_for_recipe(config, {})

    assert model_config.assistant_id == config.get(ASSISTANT_ID_KEY)
