from typing import Text, Dict, Any

import pytest

import rasa.shared.utils.io
from rasa.core.policies.ted_policy import TEDPolicyGraphComponent
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
from rasa.nlu.classifiers.mitie_intent_classifier import (
    MitieIntentClassifierGraphComponent,
)
from rasa.nlu.classifiers.sklearn_intent_classifier import (
    SklearnIntentClassifierGraphComponent,
)
from rasa.nlu.extractors.mitie_entity_extractor import (
    MitieEntityExtractorGraphComponent,
)
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.importers.autoconfig import TrainingType
import rasa.engine.validation
from rasa.shared.importers.rasa import RasaFileImporter


def test_recipe_for_name():
    recipe = Recipe.recipe_for_name("default.v1")
    assert isinstance(recipe, DefaultV1Recipe)


@pytest.mark.parametrize(
    "config_path, expected_train_schema_path, expected_predict_schema_path, "
    "training_type, is_finetuning",
    [
        # The default config is the config which most users run
        (
            "rasa/shared/importers/default_config.yml",
            "data/graph_schemas/default_config_e2e_train_schema.yml",
            "data/graph_schemas/default_config_e2e_predict_schema.yml",
            TrainingType.END_TO_END,
            False,
        ),
        # The default config without end to end
        (
            "rasa/shared/importers/default_config.yml",
            "data/graph_schemas/default_config_train_schema.yml",
            "data/graph_schemas/default_config_predict_schema.yml",
            TrainingType.BOTH,
            False,
        ),
        (
            "rasa/shared/importers/default_config.yml",
            "data/graph_schemas/default_config_core_train_schema.yml",
            "data/graph_schemas/default_config_core_predict_schema.yml",
            TrainingType.CORE,
            False,
        ),
        (
            "rasa/shared/importers/default_config.yml",
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
            "rasa/shared/importers/default_config.yml",
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

    recipe = Recipe.recipe_for_name(DefaultV1Recipe.name,)
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
    model_config = recipe.graph_config_for_recipe(config, {},)

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
        config, {"augmentation": augmentation, "debug_plots": debug_plots},
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
        config, {"persist_nlu_training_data": True},
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
            (
                SklearnIntentClassifierGraphComponent,
                MitieEntityExtractorGraphComponent,
                MitieIntentClassifierGraphComponent,
            ),
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
        "rasa/shared/importers/default_config.yml"
    )

    recipe = Recipe.recipe_for_name(DefaultV1Recipe.name)
    model_config = recipe.graph_config_for_recipe(
        config, {"finetuning_epoch_fraction": 0.5}, is_finetuning=True
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
        {
            "policies": [
                {"name": "rasa.core.policies.ted_policy.TEDPolicyGraphComponent"}
            ]
        },
        {},
        TrainingType.CORE,
    )

    assert any(
        issubclass(node.uses, TEDPolicyGraphComponent)
        for node in model_config.train_schema.nodes.values()
    )
    assert any(
        issubclass(node.uses, TEDPolicyGraphComponent)
        for node in model_config.predict_schema.nodes.values()
    )


def test_retrieve_via_invalid_module_path():
    with pytest.raises(ImportError):
        path = "rasa.core.policies.ted_policy.TEDPolicyGraphComponent1000"
        DefaultV1Recipe().graph_config_for_recipe(
            {"policies": [{"name": path}]}, {}, TrainingType.CORE,
        )


def test_train_nlu_without_nlu_pipeline():
    with pytest.raises(InvalidConfigException):
        DefaultV1Recipe().graph_config_for_recipe(
            {"pipeline": []}, {}, TrainingType.NLU,
        )


def test_train_core_without_nlu_pipeline():
    with pytest.raises(InvalidConfigException):
        DefaultV1Recipe().graph_config_for_recipe(
            {"policies": []}, {}, TrainingType.CORE,
        )
