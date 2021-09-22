from __future__ import annotations

import copy
import enum
import logging
import math
from enum import Enum
from typing import Dict, Text, Any, Tuple, Type, Optional, List, Callable

import dataclasses

from rasa.core.featurizers.precomputation import (
    CoreFeaturizationInputConverter,
    CoreFeaturizationCollector,
)
from rasa.core.policies.ensemble import DefaultPolicyPredictionEnsemble

from rasa.engine.graph import (
    GraphSchema,
    GraphComponent,
    SchemaNode,
    ExecutionContext,
    PLACEHOLDER_IMPORTER,
    PLACEHOLDER_MESSAGE,
    PLACEHOLDER_TRACKER,
)
from rasa.engine.recipes.recipe import Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.adders.nlu_prediction_to_history_adder import (
    NLUPredictionToHistoryAdder,
)
from rasa.graph_components.converters.nlu_message_converter import NLUMessageConverter
from rasa.graph_components.providers.domain_provider import DomainProvider
from rasa.graph_components.providers.domain_without_response_provider import (
    DomainWithoutResponsesProvider,
)
from rasa.graph_components.providers.nlu_training_data_provider import (
    NLUTrainingDataProvider,
)
from rasa.graph_components.providers.rule_only_provider import RuleOnlyDataProvider
from rasa.graph_components.providers.story_graph_provider import StoryGraphProvider
from rasa.graph_components.providers.training_tracker_provider import (
    TrainingTrackerProvider,
)
from rasa.graph_components.validators.finetuning_validator import FinetuningValidator

from rasa.shared.exceptions import RasaException, InvalidConfigException
from rasa.shared.importers.autoconfig import TrainingType

from rasa.shared.importers.importer import TrainingDataImporter
from rasa.utils.tensorflow.constants import EPOCHS
import rasa.shared.utils.common

logger = logging.getLogger(__name__)


class SchemaValidator(GraphComponent):
    """Temporary placeholder."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates component."""
        pass

    def validate(self, importer: TrainingDataImporter) -> TrainingDataImporter:
        """Validates component."""
        return importer


default_predict_kwargs = dict(constructor_name="load", eager=True, is_target=False,)


class DefaultV1RecipeRegisterException(RasaException):
    """If you register a class which is not of type `GraphComponent`."""

    pass


class DefaultV1Recipe(Recipe):
    """Recipe which converts the normal model config to train and predict graph."""

    @enum.unique
    class ComponentType(Enum):
        """Enum to categorize and place custom components correctly in the graph."""

        MESSAGE_TOKENIZER = 0
        MESSAGE_FEATURIZER = 1
        INTENT_CLASSIFIER = 2
        ENTITY_EXTRACTOR = 3
        POLICY_WITHOUT_END_TO_END_SUPPORT = 4
        POLICY_WITH_END_TO_END_SUPPORT = 5
        MODEL_LOADER = 6

    name = "default.v1"
    _registered_components: Dict[Text, RegisteredComponent] = {}

    def __init__(self) -> None:
        """Creates recipe."""
        self._use_core = True
        self._use_nlu = True
        self._use_end_to_end = True
        self._is_finetuning = False

        from rasa.nlu.classifiers.diet_classifier import (  # noqa: F401
            DIETClassifierGraphComponent,
        )
        from rasa.nlu.classifiers.fallback_classifier import (  # noqa: F401
            FallbackClassifierGraphComponent,
        )
        from rasa.nlu.classifiers.keyword_intent_classifier import (  # noqa: F401
            KeywordIntentClassifierGraphComponent,
        )
        from rasa.nlu.classifiers.mitie_intent_classifier import (  # noqa: F401
            MitieIntentClassifierGraphComponent,
        )
        from rasa.nlu.classifiers.sklearn_intent_classifier import (  # noqa: F401
            SklearnIntentClassifierGraphComponent,
        )
        from rasa.nlu.extractors.crf_entity_extractor import (  # noqa: F401
            CRFEntityExtractorGraphComponent,
        )
        from rasa.nlu.extractors.duckling_entity_extractor import (  # noqa: F401
            DucklingEntityExtractorGraphComponent,
        )
        from rasa.nlu.extractors.entity_synonyms import (  # noqa: F401
            EntitySynonymMapperGraphComponent,
        )
        from rasa.nlu.extractors.mitie_entity_extractor import (  # noqa: F401
            MitieEntityExtractorGraphComponent,
        )
        from rasa.nlu.extractors.spacy_entity_extractor import (  # noqa: F401
            SpacyEntityExtractorGraphComponent,
        )
        from rasa.nlu.extractors.regex_entity_extractor import (  # noqa: F401
            RegexEntityExtractorGraphComponent,
        )
        from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (  # noqa: F401, E501
            LexicalSyntacticFeaturizerGraphComponent,
        )
        from rasa.nlu.featurizers.dense_featurizer.convert_featurizer import (  # noqa: F401, E501
            ConveRTFeaturizerGraphComponent,
        )
        from rasa.nlu.featurizers.dense_featurizer.mitie_featurizer import (  # noqa: F401, E501
            MitieFeaturizerGraphComponent,
        )
        from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import (  # noqa: F401, E501
            SpacyFeaturizerGraphComponent,
        )
        from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (  # noqa: F401, E501
            CountVectorsFeaturizerGraphComponent,
        )
        from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import (  # noqa: F401
            LanguageModelFeaturizerGraphComponent,
        )
        from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import (  # noqa: F401, E501
            RegexFeaturizerGraphComponent,
        )
        from rasa.nlu.selectors.response_selector import (  # noqa: F401
            ResponseSelectorGraphComponent,
        )
        from rasa.nlu.tokenizers.jieba_tokenizer import (  # noqa: F401
            JiebaTokenizerGraphComponent,
        )
        from rasa.nlu.tokenizers.mitie_tokenizer import (  # noqa: F401
            MitieTokenizerGraphComponent,
        )
        from rasa.nlu.tokenizers.spacy_tokenizer import (  # noqa: F401
            SpacyTokenizerGraphComponent,
        )
        from rasa.nlu.tokenizers.whitespace_tokenizer import (  # noqa: F401
            WhitespaceTokenizerGraphComponent,
        )
        from rasa.nlu.utils.mitie_utils import MitieNLPGraphComponent  # noqa: F401
        from rasa.nlu.utils.spacy_utils import SpacyNLPGraphComponent  # noqa: F401

        from rasa.core.policies.ted_policy import TEDPolicyGraphComponent  # noqa: F401
        from rasa.core.policies.memoization import (  # noqa: F401
            MemoizationPolicyGraphComponent,
            AugmentedMemoizationPolicyGraphComponent,
        )
        from rasa.core.policies.rule_policy import (  # noqa: F401
            RulePolicyGraphComponent,
        )
        from rasa.core.policies.unexpected_intent_policy import (  # noqa: F401
            UnexpecTEDIntentPolicyGraphComponent,
        )

    @dataclasses.dataclass()
    class RegisteredComponent:
        """Describes a graph component which was registered with the decorator."""

        clazz: Type[GraphComponent]
        type: DefaultV1Recipe.ComponentType
        is_trainable: bool
        model_from: Optional[Text]

    @classmethod
    def register(
        cls,
        component_type: ComponentType,
        is_trainable: bool,
        model_from: Optional[Text] = None,
    ) -> Callable[[Type[GraphComponent]], Type[GraphComponent]]:
        """This decorator can be used to register classes with the recipe.

        Args:
            component_type: Describes the type of the component which is then used
                to place the component in the graph.
            is_trainable: `True` if the component requires training.
            model_from: Can be used if this component requires a pre-loaded model
                such as `SpacyNLP` or `MitieNLP`.

        Returns:
            The registered class.
        """

        def decorator(registered_class: Type[GraphComponent]) -> Type[GraphComponent]:
            if not issubclass(registered_class, GraphComponent):
                raise DefaultV1RecipeRegisterException(
                    f"Failed to register class '{registered_class.__name__}' with "
                    f"the recipe '{cls.name}'. The class has to be of type "
                    f"'{GraphComponent.__name__}'."
                )
            cls._registered_components[
                registered_class.__name__
            ] = cls.RegisteredComponent(
                registered_class, component_type, is_trainable, model_from
            )
            return registered_class

        return decorator

    @classmethod
    def _from_registry(cls, name: Text) -> RegisteredComponent:
        # TODO: Hack until we've deleted the old components
        if not name.endswith("GraphComponent"):
            name = f"{name}GraphComponent"

        if name in cls._registered_components:
            return cls._registered_components[name]

        if "." in name:
            clazz = rasa.shared.utils.common.class_from_module_path(name)
            if clazz.__name__ in cls._registered_components:
                return cls._registered_components[clazz.__name__]

        raise InvalidConfigException(
            f"Can't load class for name '{name}'. Please make sure to provide "
            f"a valid name or module path and to register it using the "
            f"'@DefaultV1Recipe.register' decorator."
        )

    def schemas_for_config(
        self,
        config: Dict,
        cli_parameters: Dict[Text, Any],
        training_type: TrainingType = TrainingType.BOTH,
        is_finetuning: bool = False,
    ) -> Tuple[GraphSchema, GraphSchema]:
        """Converts the default config to graphs (see interface for full docstring)."""
        self._use_core = (
            bool(config.get("policies")) and not training_type == TrainingType.NLU
        )
        self._use_nlu = (
            bool(config.get("pipeline")) and not training_type == TrainingType.CORE
        )

        if not self._use_nlu and training_type == TrainingType.NLU:
            raise InvalidConfigException(
                "Can't train an NLU model without a specified pipeline. Please make "
                "sure to specify a valid pipeline in your configuration."
            )

        if not self._use_core and training_type == TrainingType.CORE:
            raise InvalidConfigException(
                "Can't train an Core model without policies. Please make "
                "sure to specify a valid policy in your configuration."
            )

        self._use_end_to_end = (
            self._use_nlu
            and self._use_core
            and training_type == TrainingType.END_TO_END
        )

        self._is_finetuning = is_finetuning

        train_nodes, preprocessors = self._create_train_nodes(config, cli_parameters)
        predict_nodes = self._create_predict_nodes(config, preprocessors, train_nodes)

        return GraphSchema(train_nodes), GraphSchema(predict_nodes)

    def _create_train_nodes(
        self, config: Dict[Text, Any], cli_parameters: Dict[Text, Any]
    ) -> Tuple[Dict[Text, SchemaNode], List[Text]]:
        train_config = copy.deepcopy(config)

        train_nodes = {
            "schema_validator": SchemaNode(
                needs={"importer": PLACEHOLDER_IMPORTER},
                uses=SchemaValidator,
                constructor_name="create",
                fn="validate",
                config={},
                is_input=True,
            ),
            "finetuning_validator": SchemaNode(
                needs={"importer": "schema_validator"},
                uses=FinetuningValidator,
                constructor_name="load" if self._is_finetuning else "create",
                fn="validate",
                is_input=True,
                config={"validate_core": self._use_core, "validate_nlu": self._use_nlu},
            ),
        }

        preprocessors = []

        if self._use_nlu:
            preprocessors = self._add_nlu_train_nodes(
                train_config, train_nodes, cli_parameters
            )

        if self._use_core:
            self._add_core_train_nodes(
                train_config, train_nodes, preprocessors, cli_parameters
            )

        return train_nodes, preprocessors

    def _add_nlu_train_nodes(
        self,
        train_config: Dict[Text, Any],
        train_nodes: Dict[Text, SchemaNode],
        cli_parameters: Dict[Text, Any],
    ) -> List[Text]:
        persist_nlu_data = bool(cli_parameters.get("persist_nlu_training_data"))
        train_nodes["nlu_training_data_provider"] = SchemaNode(
            needs={"importer": "finetuning_validator"},
            uses=NLUTrainingDataProvider,
            constructor_name="create",
            fn="provide",
            config={
                "language": train_config.get("language"),
                "persist": persist_nlu_data,
            },
            is_target=persist_nlu_data,
            is_input=True,
        )

        last_run_node = "nlu_training_data_provider"
        preprocessors: List[Text] = []

        for idx, item in enumerate(train_config["pipeline"]):
            component_name = item.pop("name")
            component = self._from_registry(component_name)
            component_name = f"{component_name}{idx}"

            if component.type == self.ComponentType.MODEL_LOADER:
                node_name = f"run_{component_name}"
                train_nodes[node_name] = SchemaNode(
                    needs={},
                    uses=component.clazz,
                    constructor_name="create",
                    fn="provide",
                    config=item,
                )
                # TODO: Spacy preprocessor
            else:
                from_resource = None
                if component.is_trainable:
                    from_resource = self._add_nlu_train_node(
                        train_nodes,
                        component.clazz,
                        component_name,
                        last_run_node,
                        item,
                        cli_parameters,
                    )

                if component.type in [
                    self.ComponentType.MESSAGE_TOKENIZER,
                    self.ComponentType.MESSAGE_FEATURIZER,
                ]:
                    last_run_node = self._add_nlu_process_node(
                        train_nodes,
                        component.clazz,
                        component_name,
                        last_run_node,
                        item,
                        from_resource=from_resource,
                    )

                    # Remember for End-to-End-Featurization
                    preprocessors.append(last_run_node)

        return preprocessors

    def _add_nlu_train_node(
        self,
        train_nodes: Dict[Text, SchemaNode],
        component: Type[GraphComponent],
        component_name: Text,
        last_run_node: Text,
        config: Dict[Text, Any],
        cli_parameters: Dict[Text, Any],
    ) -> Text:
        config_from_cli = self._extra_config_from_cli(cli_parameters, component, config)
        model_provider_needs = self._get_model_provider_needs(train_nodes, component,)

        train_node_name = f"train_{component_name}"
        train_nodes[train_node_name] = SchemaNode(
            needs={"training_data": last_run_node, **model_provider_needs},
            uses=component,
            constructor_name="load" if self._is_finetuning else "create",
            fn="train",
            config={**config, **config_from_cli},
            is_target=True,
        )
        return train_node_name

    def _extra_config_from_cli(
        self,
        cli_parameters: Dict[Text, Any],
        component: Type[GraphComponent],
        component_config: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        from rasa.nlu.classifiers.mitie_intent_classifier import (
            MitieIntentClassifierGraphComponent,
        )
        from rasa.nlu.extractors.mitie_entity_extractor import (
            MitieEntityExtractorGraphComponent,
        )
        from rasa.nlu.classifiers.sklearn_intent_classifier import (
            SklearnIntentClassifierGraphComponent,
        )

        cli_args_mapping: Dict[Type[GraphComponent], List[Text]] = {
            MitieIntentClassifierGraphComponent: ["num_threads"],
            MitieEntityExtractorGraphComponent: ["num_threads"],
            SklearnIntentClassifierGraphComponent: ["num_threads"],
        }

        config_from_cli = {
            param: cli_parameters[param]
            for param in cli_args_mapping.get(component, [])
            if param in cli_parameters
        }

        if (
            self._is_finetuning
            and "finetuning_epoch_fraction" in cli_parameters
            and EPOCHS in component.get_default_config()
        ):
            old_number_epochs = component_config.get(
                EPOCHS, component.get_default_config()[EPOCHS]
            )
            epoch_fraction = float(cli_parameters["finetuning_epoch_fraction"])

            config_from_cli[EPOCHS] = math.ceil(old_number_epochs * epoch_fraction)

        return config_from_cli

    def _add_nlu_process_node(
        self,
        train_nodes: Dict[Text, SchemaNode],
        component_class: Type[GraphComponent],
        component_name: Text,
        last_run_node: Text,
        component_config: Dict[Text, Any],
        from_resource: Optional[Text] = None,
    ) -> Text:
        resource_needs = {}
        if from_resource:
            resource_needs = {"resource": from_resource}

        model_provider_needs = self._get_model_provider_needs(
            train_nodes, component_class,
        )

        node_name = f"run_{component_name}"
        train_nodes[node_name] = SchemaNode(
            needs={
                "training_data": last_run_node,
                **resource_needs,
                **model_provider_needs,
            },
            uses=component_class,
            constructor_name="load",
            fn="process_training_data",
            config=component_config,
        )
        return node_name

    def _get_model_provider_needs(
        self, nodes: Dict[Text, SchemaNode], component_class: Type[GraphComponent],
    ) -> Dict[Text, Text]:
        model_provider_needs = {}
        component = self._from_registry(component_class.__name__)

        if not component.model_from:
            return {}

        node_name_of_provider = next(
            (
                node_name
                for node_name, node in nodes.items()
                if node.uses.__name__ == component.model_from
            ),
            None,
        )
        if node_name_of_provider:
            model_provider_needs["model"] = node_name_of_provider

        return model_provider_needs

    def _add_core_train_nodes(
        self,
        train_config: Dict[Text, Any],
        train_nodes: Dict[Text, SchemaNode],
        preprocessors: List[Text],
        cli_parameters: Dict[Text, Any],
    ) -> None:
        train_nodes["domain_provider"] = SchemaNode(
            needs={"importer": "finetuning_validator"},
            uses=DomainProvider,
            constructor_name="create",
            fn="provide_train",
            config={},
            is_target=True,
            is_input=True,
        )
        train_nodes["domain_without_responses_provider"] = SchemaNode(
            needs={"domain": "domain_provider"},
            uses=DomainWithoutResponsesProvider,
            constructor_name="create",
            fn="provide",
            config={},
            is_input=True,
        )
        train_nodes["story_graph_provider"] = SchemaNode(
            needs={"importer": "finetuning_validator"},
            uses=StoryGraphProvider,
            constructor_name="create",
            fn="provide",
            config={"exclusion_percentage": cli_parameters.get("exclusion_percentage")},
            is_input=True,
        )
        train_nodes["training_tracker_provider"] = SchemaNode(
            needs={
                "story_graph": "story_graph_provider",
                "domain": "domain_without_responses_provider",
            },
            uses=TrainingTrackerProvider,
            constructor_name="create",
            fn="provide",
            config={
                config_key: cli_parameters[param]
                for param, config_key in {
                    "debug_plots": "debug_plots",
                    "augmentation": "augmentation_factor",
                }.items()
                if param in cli_parameters
            },
        )

        policy_with_end_to_end_support_used = False
        for idx, item in enumerate(train_config["policies"]):
            component_name = item.pop("name")
            component = self._from_registry(component_name)

            extra_config_from_cli = self._extra_config_from_cli(
                cli_parameters, component.clazz, item
            )

            requires_end_to_end_data = self._use_end_to_end and (
                component.type == self.ComponentType.POLICY_WITH_END_TO_END_SUPPORT
            )
            policy_with_end_to_end_support_used = (
                policy_with_end_to_end_support_used or requires_end_to_end_data
            )

            train_nodes[f"train_{component_name}{idx}"] = SchemaNode(
                needs={
                    "training_trackers": "training_tracker_provider",
                    "domain": "domain_without_responses_provider",
                    **(
                        {"precomputations": "end_to_end_features_provider"}
                        if requires_end_to_end_data
                        else {}
                    ),
                },
                uses=component.clazz,
                constructor_name="load" if self._is_finetuning else "create",
                fn="train",
                is_target=True,
                config={**item, **extra_config_from_cli},
            )

        if self._use_end_to_end and policy_with_end_to_end_support_used:
            self._add_end_to_end_features_for_training(preprocessors, train_nodes)

    def _add_end_to_end_features_for_training(
        self, preprocessors: List[Text], train_nodes: Dict[Text, SchemaNode],
    ) -> None:
        train_nodes["story_to_nlu_training_data_converter"] = SchemaNode(
            needs={
                "story_graph": "story_graph_provider",
                "domain": "domain_without_responses_provider",
            },
            uses=CoreFeaturizationInputConverter,
            constructor_name="create",
            fn="convert_for_training",
            config={},
            is_input=True,
        )

        last_node_name = "story_to_nlu_training_data_converter"
        for preprocessor in preprocessors:
            node = copy.deepcopy(train_nodes[preprocessor])
            node.needs["training_data"] = last_node_name

            node_name = f"e2e_{preprocessor}"
            train_nodes[node_name] = node
            last_node_name = node_name

        node_with_e2e_features = "end_to_end_features_provider"
        train_nodes[node_with_e2e_features] = SchemaNode(
            needs={"messages": last_node_name,},
            uses=CoreFeaturizationCollector,
            constructor_name="create",
            fn="collect",
            config={},
        )

    def _create_predict_nodes(
        self,
        config: Dict[Text, SchemaNode],
        preprocessors: List[Text],
        train_nodes: Dict[Text, SchemaNode],
    ) -> Dict[Text, SchemaNode]:

        predict_config = copy.deepcopy(config)
        predict_nodes = {}

        nlu_output_node = None

        if self._use_nlu:
            nlu_output_node = self._add_nlu_predict_nodes(
                predict_config, predict_nodes, train_nodes
            )

        if self._use_core:
            self._add_core_predict_nodes(
                predict_config,
                predict_nodes,
                nlu_output_node,
                train_nodes,
                preprocessors,
            )

        return predict_nodes

    def _add_nlu_predict_nodes(
        self,
        predict_config: Dict[Text, Any],
        predict_nodes: Dict[Text, SchemaNode],
        train_nodes: Dict[Text, SchemaNode],
    ) -> Text:
        predict_nodes["nlu_message_converter"] = SchemaNode(
            **default_predict_kwargs,
            needs={"messages": PLACEHOLDER_MESSAGE},
            uses=NLUMessageConverter,
            fn="convert_user_message",
            config={},
        )

        last_run_node = "nlu_message_converter"

        for idx, item in enumerate(predict_config["pipeline"]):
            component_name = item.pop("name")
            component = self._from_registry(component_name)
            component_name = f"{component_name}{idx}"
            if component.type == self.ComponentType.MODEL_LOADER:
                predict_nodes[f"run_{component_name}"] = SchemaNode(
                    **default_predict_kwargs,
                    needs={},
                    uses=component.clazz,
                    fn="provide",
                    config=item,
                )
            elif component.type in [
                self.ComponentType.MESSAGE_TOKENIZER,
                self.ComponentType.MESSAGE_FEATURIZER,
            ]:
                last_run_node = self._add_nlu_predict_node_from_train(
                    predict_nodes,
                    component_name,
                    train_nodes,
                    last_run_node,
                    item,
                    from_resource=component.is_trainable,
                )
            elif component.type in [
                self.ComponentType.INTENT_CLASSIFIER,
                self.ComponentType.ENTITY_EXTRACTOR,
            ]:
                if component.is_trainable:
                    last_run_node = self._add_nlu_predict_node_from_train(
                        predict_nodes,
                        component_name,
                        train_nodes,
                        last_run_node,
                        item,
                        from_resource=component.is_trainable,
                    )
                else:
                    new_node = SchemaNode(
                        needs={"messages": last_run_node},
                        uses=component.clazz,
                        constructor_name="create",
                        fn="process",
                        config=item,
                    )

                    last_run_node = self._add_nlu_predict_node(
                        predict_nodes, new_node, component_name, last_run_node
                    )

        from rasa.nlu.classifiers.regex_message_handler import (
            RegexMessageHandlerGraphComponent,
        )

        node_name = f"run_{RegexMessageHandlerGraphComponent.__name__}"

        domain_needs = {}
        if self._use_core:
            domain_needs["domain"] = "domain_provider"
        predict_nodes[node_name] = SchemaNode(
            **default_predict_kwargs,
            needs={"messages": last_run_node, **domain_needs},
            uses=RegexMessageHandlerGraphComponent,
            fn="process",
            config={},
        )

        return node_name

    def _add_nlu_predict_node_from_train(
        self,
        predict_nodes: Dict[Text, SchemaNode],
        node_name: Text,
        train_nodes: Dict[Text, SchemaNode],
        last_run_node: Text,
        item_config: Dict[Text, Any],
        from_resource: bool = False,
    ) -> Text:
        train_node_name = f"run_{node_name}"
        resource = None
        if from_resource:
            train_node_name = f"train_{node_name}"
            resource = Resource(train_node_name)

        return self._add_nlu_predict_node(
            predict_nodes,
            dataclasses.replace(
                train_nodes[train_node_name], resource=resource, config=item_config
            ),
            node_name,
            last_run_node,
        )

    def _add_nlu_predict_node(
        self,
        predict_nodes: Dict[Text, SchemaNode],
        node: SchemaNode,
        component_name: Text,
        last_run_node: Text,
    ) -> Text:
        node_name = f"run_{component_name}"
        model_provider_needs = self._get_model_provider_needs(predict_nodes, node.uses,)

        predict_nodes[node_name] = dataclasses.replace(
            node,
            needs={"messages": last_run_node, **model_provider_needs},
            fn="process",
            **default_predict_kwargs,
        )

        return node_name

    def _add_core_predict_nodes(
        self,
        predict_config: Dict[Text, Any],
        predict_nodes: Dict[Text, SchemaNode],
        nlu_output_node: Optional[Text],
        train_nodes: Dict[Text, SchemaNode],
        preprocessors: List[Text],
    ) -> None:
        if nlu_output_node:
            predict_nodes["nlu_prediction_to_history_adder"] = SchemaNode(
                **default_predict_kwargs,
                needs={
                    "predictions": nlu_output_node,
                    "domain": "domain_provider",
                    "original_messages": PLACEHOLDER_MESSAGE,
                    "tracker": PLACEHOLDER_TRACKER,
                },
                uses=NLUPredictionToHistoryAdder,
                fn="add",
                config={},
            )
        predict_nodes["domain_provider"] = SchemaNode(
            **default_predict_kwargs,
            needs={},
            uses=DomainProvider,
            fn="provide_inference",
            config={},
            resource=Resource("domain_provider"),
        )

        node_with_e2e_features = None

        if "end_to_end_features_provider" in train_nodes:
            node_with_e2e_features = self._add_end_to_end_features_for_inference(
                predict_nodes, preprocessors
            )

        rule_only_data_provider_name = "rule_only_data_provider"
        rule_policy_resource = None
        policies: List[Text] = []

        for idx, item in enumerate(predict_config["policies"]):
            component_name = item.pop("name")
            component = self._from_registry(component_name)

            train_node_name = f"train_{component_name}{idx}"
            node_name = f"run_{component_name}{idx}"

            from rasa.core.policies.rule_policy import RulePolicyGraphComponent

            if issubclass(component.clazz, RulePolicyGraphComponent):
                rule_policy_resource = train_node_name

            predict_nodes[node_name] = dataclasses.replace(
                train_nodes[train_node_name],
                **default_predict_kwargs,
                needs={
                    "domain": "domain_provider",
                    **(
                        {"precomputations": node_with_e2e_features}
                        if component.type
                        == self.ComponentType.POLICY_WITH_END_TO_END_SUPPORT
                        and node_with_e2e_features
                        else {}
                    ),
                    "tracker": "nlu_prediction_to_history_adder"
                    if self._use_nlu
                    else PLACEHOLDER_TRACKER,
                    "rule_only_data": rule_only_data_provider_name,
                },
                fn="predict_action_probabilities",
                resource=Resource(train_node_name),
            )
            policies.append(node_name)

        predict_nodes["rule_only_data_provider"] = SchemaNode(
            **default_predict_kwargs,
            needs={},
            uses=RuleOnlyDataProvider,
            fn="provide",
            config={},
            resource=Resource(rule_policy_resource) if rule_policy_resource else None,
        )

        predict_nodes["select_prediction"] = SchemaNode(
            **default_predict_kwargs,
            needs={
                **{f"policy{idx}": name for idx, name in enumerate(policies)},
                "domain": "domain_provider",
                "tracker": "nlu_prediction_to_history_adder"
                if self._use_nlu
                else PLACEHOLDER_TRACKER,
            },
            uses=DefaultPolicyPredictionEnsemble,
            fn="combine_predictions_from_kwargs",
            config={},
        )

    def _add_end_to_end_features_for_inference(
        self, predict_nodes: Dict[Text, SchemaNode], preprocessors: List[Text],
    ) -> Text:
        predict_nodes["tracker_to_message_converter"] = SchemaNode(
            **default_predict_kwargs,
            needs={"tracker": "nlu_prediction_to_history_adder"},
            uses=CoreFeaturizationInputConverter,
            fn="convert_for_inference",
            config={},
        )

        last_node_name = "tracker_to_message_converter"
        for preprocessor in preprocessors:
            node = dataclasses.replace(
                predict_nodes[preprocessor], needs={"messages": last_node_name}
            )

            node_name = f"e2e_{preprocessor}"
            predict_nodes[node_name] = node
            last_node_name = node_name

        node_with_e2e_features = "end_to_end_features_provider"
        predict_nodes[node_with_e2e_features] = SchemaNode(
            **default_predict_kwargs,
            needs={"messages": last_node_name,},
            uses=CoreFeaturizationCollector,
            fn="collect",
            config={},
        )
        return node_with_e2e_features
