from __future__ import annotations

import copy
import enum
import logging
import math
from enum import Enum
from typing import Dict, Text, Any, Tuple, Type, Optional, List, Callable, Set, Union

import dataclasses

from rasa.core.featurizers.precomputation import (
    CoreFeaturizationInputConverter,
    CoreFeaturizationCollector,
)
from rasa.graph_components.providers.flows_provider import FlowsProvider
from rasa.dialogue_understanding.processor.command_processor_component import (
    CommandProcessorComponent,
)
from rasa.shared.exceptions import FileNotFoundException
from rasa.core.policies.ensemble import DefaultPolicyPredictionEnsemble

from rasa.engine.graph import (
    GraphSchema,
    GraphComponent,
    SchemaNode,
    GraphModelConfiguration,
)
from rasa.engine.constants import (
    PLACEHOLDER_IMPORTER,
    PLACEHOLDER_MESSAGE,
    PLACEHOLDER_TRACKER,
    PLACEHOLDER_ENDPOINTS,
)
from rasa.engine.recipes.recipe import Recipe
from rasa.engine.storage.resource import Resource
from rasa.graph_components.converters.nlu_message_converter import NLUMessageConverter
from rasa.graph_components.providers.domain_provider import DomainProvider
from rasa.graph_components.providers.forms_provider import FormsProvider
from rasa.graph_components.providers.responses_provider import ResponsesProvider
from rasa.graph_components.providers.domain_for_core_training_provider import (
    DomainForCoreTrainingProvider,
)
from rasa.graph_components.providers.nlu_training_data_provider import (
    NLUTrainingDataProvider,
)
from rasa.graph_components.providers.rule_only_provider import RuleOnlyDataProvider
from rasa.graph_components.providers.story_graph_provider import StoryGraphProvider
from rasa.graph_components.providers.training_tracker_provider import (
    TrainingTrackerProvider,
)
import rasa.shared.constants
from rasa.shared.exceptions import RasaException, InvalidConfigException
from rasa.shared.constants import ASSISTANT_ID_KEY
from rasa.shared.data import TrainingType
from rasa.shared.utils.yaml import read_config_file

from rasa.utils.tensorflow.constants import EPOCHS
from rasa.shared.utils.common import (
    class_from_module_path,
    transform_collection_to_sentence,
)

logger = logging.getLogger(__name__)


DEFAULT_PREDICT_KWARGS = dict(constructor_name="load", eager=True, is_target=False)

COMMENTS_FOR_KEYS = {
    "pipeline": (
        f"# # No configuration for the NLU pipeline was provided. The following "
        f"default pipeline was used to train your model.\n"
        f"# # If you'd like to customize it, uncomment and adjust the pipeline.\n"
        f"# # See {rasa.shared.constants.DOCS_URL_PIPELINE} for more information.\n"
    ),
    "policies": (
        f"# # No configuration for policies was provided. The following default "
        f"policies were used to train your model.\n"
        f"# # If you'd like to customize them, uncomment and adjust the policies.\n"
        f"# # See {rasa.shared.constants.DOCS_URL_POLICIES} for more information.\n"
    ),
}


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
        COMMAND_GENERATOR = 7
        COEXISTENCE_ROUTER = 8

    name = "default.v1"
    _registered_components: Dict[Text, RegisteredComponent] = {}  # noqa: RUF012

    def __init__(self) -> None:
        """Creates recipe."""
        self._use_core = True
        self._use_nlu = True
        self._use_end_to_end = True
        self._is_finetuning = False

    @dataclasses.dataclass()
    class RegisteredComponent:
        """Describes a graph component which was registered with the decorator."""

        clazz: Type[GraphComponent]
        types: Set[DefaultV1Recipe.ComponentType]
        is_trainable: bool
        model_from: Optional[Text]

    @classmethod
    def register(
        cls,
        component_types: Union[ComponentType, List[ComponentType]],
        is_trainable: bool,
        model_from: Optional[Text] = None,
    ) -> Callable[[Type[GraphComponent]], Type[GraphComponent]]:
        """This decorator can be used to register classes with the recipe.

        Args:
            component_types: Describes the types of a component which are then used
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

            if isinstance(component_types, cls.ComponentType):
                unique_types = {component_types}
            else:
                unique_types = set(component_types)

            cls._registered_components[registered_class.__name__] = (
                cls.RegisteredComponent(
                    registered_class, unique_types, is_trainable, model_from
                )
            )
            return registered_class

        return decorator

    @classmethod
    def _from_registry(cls, name: Text) -> RegisteredComponent:
        # Importing all the default Rasa components will automatically register them
        from rasa.engine.recipes.default_components import DEFAULT_COMPONENTS  # noqa

        if name in cls._registered_components:
            return cls._registered_components[name]

        if "." in name:
            clazz = class_from_module_path(name)
            if clazz.__name__ in cls._registered_components:
                return cls._registered_components[clazz.__name__]

        raise InvalidConfigException(
            f"Can't load class for name '{name}'. Please make sure to provide "
            f"a valid name or module path and to register it using the "
            f"'@DefaultV1Recipe.register' decorator."
        )

    def graph_config_for_recipe(
        self,
        config: Dict,
        cli_parameters: Dict[Text, Any],
        training_type: TrainingType = TrainingType.BOTH,
        is_finetuning: bool = False,
    ) -> GraphModelConfiguration:
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

        core_target = "select_prediction" if self._use_core else None

        from rasa.nlu.classifiers.regex_message_handler import RegexMessageHandler

        return GraphModelConfiguration(
            train_schema=GraphSchema(train_nodes),
            predict_schema=GraphSchema(predict_nodes),
            training_type=training_type,
            assistant_id=config.get(ASSISTANT_ID_KEY),
            language=config.get("language"),
            spaces=config.get("spaces"),
            core_target=core_target,
            nlu_target=f"run_{RegexMessageHandler.__name__}",
        )

    def _create_train_nodes(
        self, config: Dict[Text, Any], cli_parameters: Dict[Text, Any]
    ) -> Tuple[Dict[Text, SchemaNode], List[Text]]:
        from rasa.graph_components.validators.default_recipe_validator import (
            DefaultV1RecipeValidator,
        )
        from rasa.graph_components.validators.finetuning_validator import (
            FinetuningValidator,
        )

        train_config = copy.deepcopy(config)

        train_nodes = {
            "schema_validator": SchemaNode(
                needs={"importer": PLACEHOLDER_IMPORTER},
                uses=DefaultV1RecipeValidator,
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
        train_nodes["flows_provider"] = SchemaNode(
            needs={
                "importer": "finetuning_validator",
            },
            uses=FlowsProvider,
            constructor_name="create",
            fn="provide_train",
            config={},
            is_target=True,
            is_input=True,
        )
        train_nodes["domain_provider"] = SchemaNode(
            needs={
                "importer": "finetuning_validator",
            },
            uses=DomainProvider,
            constructor_name="create",
            fn="provide_train",
            config={},
            is_target=True,
            is_input=True,
        )
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

        for idx, config in enumerate(train_config["pipeline"]):
            component_name = config.pop("name")
            component = self._from_registry(component_name)
            component_name = f"{component_name}{idx}"

            if (
                self.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT in component.types
                or self.ComponentType.POLICY_WITH_END_TO_END_SUPPORT in component.types
            ):
                raise InvalidConfigException(
                    f"Found policy '{component_name}' in NLU pipeline. Policies should "
                    f"be defined in the 'policies' section of your configuration."
                )
            if self.ComponentType.MODEL_LOADER in component.types:
                node_name = f"provide_{component_name}"
                train_nodes[node_name] = SchemaNode(
                    needs={},
                    uses=component.clazz,
                    constructor_name="create",
                    fn="provide",
                    config=config,
                )

            from_resource = None
            if component.is_trainable:
                from_resource = self._add_nlu_train_node(
                    train_nodes,
                    component.clazz,
                    component_name,
                    last_run_node,
                    config,
                    cli_parameters,
                )

            if component.types.intersection(
                {
                    self.ComponentType.MESSAGE_TOKENIZER,
                    self.ComponentType.MESSAGE_FEATURIZER,
                }
            ):
                last_run_node = self._add_nlu_process_node(
                    train_nodes,
                    component.clazz,
                    component_name,
                    last_run_node,
                    config,
                    from_resource=from_resource,
                )

                # Remember for End-to-End-Featurization
                preprocessors.append(last_run_node)

        return preprocessors

    def _get_needs_from_args(
        self, component: Type[GraphComponent], fn_name: str
    ) -> Dict[str, str]:
        """Get the needed arguments from the method on the component.

        Filters out arguments that are already provided by other graph
        components. Does not check if the created providers are actually
        part of the graph. If they aren't an error will be raised later on
        when the graph is validated.

        Args:
            component: The component class.
            fn_name: The name of the method to inspect.

        Returns:
            The name of the arguments which need to be provided.
        """
        from inspect import signature

        if not hasattr(component, fn_name):
            return {}

        def resolver_name_from_parameter(parameter: str) -> str:
            # we got a couple special cases to handle where the parameter name
            # doesn't match the provider name
            if "training_trackers" == parameter:
                return "training_tracker_provider"
            elif "tracker" == parameter:
                return PLACEHOLDER_TRACKER
            elif "endpoints" == parameter:
                return PLACEHOLDER_ENDPOINTS
            elif "training_data" == parameter:
                return "nlu_training_data_provider"
            return f"{parameter}_provider"

        sig = signature(getattr(component, fn_name))
        parameters = {
            name
            for name, param in sig.parameters.items()
            if param.kind == param.POSITIONAL_OR_KEYWORD
        }

        # filter out parameters which are already resolved in other ways
        unprovided_parameters = parameters - {
            "message",
            "messages",
            "self",
            "model",
            "precomputations",
        }

        return {
            parameter: resolver_name_from_parameter(parameter)
            for parameter in unprovided_parameters
        }

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
        needs = self._get_needs_from_args(component, "train")
        needs.update(self._get_model_provider_needs(train_nodes, component))
        needs["training_data"] = last_run_node

        train_node_name = f"train_{component_name}"
        train_nodes[train_node_name] = SchemaNode(
            needs=needs,
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
        from rasa.nlu.classifiers.mitie_intent_classifier import MitieIntentClassifier
        from rasa.nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
        from rasa.nlu.classifiers.sklearn_intent_classifier import (
            SklearnIntentClassifier,
        )

        cli_args_mapping: Dict[Type[GraphComponent], List[Text]] = {
            MitieIntentClassifier: ["num_threads"],
            MitieEntityExtractor: ["num_threads"],
            SklearnIntentClassifier: ["num_threads"],
        }

        config_from_cli = {
            param: cli_parameters[param]
            for param in cli_args_mapping.get(component, [])
            if param in cli_parameters and cli_parameters[param] is not None
        }

        if (
            self._is_finetuning
            and "finetuning_epoch_fraction" in cli_parameters
            and EPOCHS in component.get_default_config()
        ):
            old_number_epochs = component_config.get(
                EPOCHS, component.get_default_config()[EPOCHS]
            )
            epoch_fraction = cli_parameters["finetuning_epoch_fraction"]
            epoch_fraction = epoch_fraction if epoch_fraction is not None else 1.0
            config_from_cli["finetuning_epoch_fraction"] = epoch_fraction
            config_from_cli[EPOCHS] = math.ceil(
                old_number_epochs * float(epoch_fraction)
            )

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
        needs = self._get_needs_from_args(component_class, "process_training_data")
        needs.update(self._get_model_provider_needs(train_nodes, component_class))

        if from_resource:
            needs["resource"] = from_resource

        needs["training_data"] = last_run_node

        node_name = f"run_{component_name}"
        train_nodes[node_name] = SchemaNode(
            needs=needs,
            uses=component_class,
            constructor_name="load",
            fn="process_training_data",
            config=component_config,
        )
        return node_name

    def _get_model_provider_needs(
        self, nodes: Dict[Text, SchemaNode], component_class: Type[GraphComponent]
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
        train_nodes["domain_for_core_training_provider"] = SchemaNode(
            needs={"domain": "domain_provider"},
            uses=DomainForCoreTrainingProvider,
            constructor_name="create",
            fn="provide",
            config={},
            is_input=True,
        )
        train_nodes["forms_provider"] = SchemaNode(
            needs={"domain": "domain_provider"},
            uses=FormsProvider,
            constructor_name="create",
            fn="provide",
            config={},
            is_input=True,
        )
        train_nodes["responses_provider"] = SchemaNode(
            needs={"domain": "domain_provider"},
            uses=ResponsesProvider,
            constructor_name="create",
            fn="provide",
            config={},
            is_input=True,
        )
        train_nodes["story_graph_provider"] = SchemaNode(
            needs={"importer": "finetuning_validator"},
            uses=StoryGraphProvider,
            constructor_name="create",
            fn="provide_train",
            config={"exclusion_percentage": cli_parameters.get("exclusion_percentage")},
            is_input=True,
        )
        train_nodes["flows_provider"] = SchemaNode(
            needs={
                "importer": "finetuning_validator",
            },
            uses=FlowsProvider,
            constructor_name="create",
            fn="provide_train",
            config={},
            is_target=True,
            is_input=True,
        )
        train_nodes["training_tracker_provider"] = SchemaNode(
            needs={
                "story_graph": "story_graph_provider",
                "domain": "domain_for_core_training_provider",
            },
            uses=TrainingTrackerProvider,
            constructor_name="create",
            fn="provide",
            config={
                param: cli_parameters[param]
                for param in ["debug_plots", "augmentation_factor"]
                if param in cli_parameters
            },
        )

        policy_with_end_to_end_support_used = False
        for idx, config in enumerate(train_config["policies"]):
            component_name = config.pop("name")
            component = self._from_registry(component_name)

            extra_config_from_cli = self._extra_config_from_cli(
                cli_parameters, component.clazz, config
            )

            requires_end_to_end_data = self._use_end_to_end and (
                self.ComponentType.POLICY_WITH_END_TO_END_SUPPORT in component.types
            )
            policy_with_end_to_end_support_used = (
                policy_with_end_to_end_support_used or requires_end_to_end_data
            )

            needs = self._get_needs_from_args(component.clazz, "train")
            if requires_end_to_end_data:
                needs["precomputations"] = "end_to_end_features_provider"
            # during core training we use a stripped down version of the domain
            needs["domain"] = "domain_for_core_training_provider"
            train_nodes[f"train_{component_name}{idx}"] = SchemaNode(
                needs=needs,
                uses=component.clazz,
                constructor_name="load" if self._is_finetuning else "create",
                fn="train",
                is_target=True,
                config={**config, **extra_config_from_cli},
            )

        if self._use_end_to_end and policy_with_end_to_end_support_used:
            self._add_end_to_end_features_for_training(preprocessors, train_nodes)

    def _add_end_to_end_features_for_training(
        self, preprocessors: List[Text], train_nodes: Dict[Text, SchemaNode]
    ) -> None:
        train_nodes["story_to_nlu_training_data_converter"] = SchemaNode(
            needs={
                "story_graph": "story_graph_provider",
                "domain": "domain_for_core_training_provider",
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
            needs={"messages": last_node_name},
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

        from rasa.nlu.classifiers.regex_message_handler import RegexMessageHandler

        predict_nodes["nlu_message_converter"] = SchemaNode(
            **DEFAULT_PREDICT_KWARGS,
            needs={"messages": PLACEHOLDER_MESSAGE},
            uses=NLUMessageConverter,
            fn="convert_user_message",
            config={},
        )

        last_run_nlu_node = "nlu_message_converter"

        if self._use_nlu:
            last_run_nlu_node = self._add_nlu_predict_nodes(
                last_run_nlu_node, predict_config, predict_nodes, train_nodes
            )

        domain_needs = {}
        if self._use_core:
            domain_needs["domain"] = "domain_provider"

        regex_handler_node_name = f"run_{RegexMessageHandler.__name__}"
        predict_nodes[regex_handler_node_name] = SchemaNode(
            **DEFAULT_PREDICT_KWARGS,
            needs={"messages": last_run_nlu_node, **domain_needs},
            uses=RegexMessageHandler,
            fn="process",
            config={},
        )

        if self._use_core:
            self._add_core_predict_nodes(
                predict_config, predict_nodes, train_nodes, preprocessors
            )

        return predict_nodes

    def _add_nlu_predict_nodes(
        self,
        last_run_node: Text,
        predict_config: Dict[Text, Any],
        predict_nodes: Dict[Text, SchemaNode],
        train_nodes: Dict[Text, SchemaNode],
    ) -> Text:
        predict_nodes["flows_provider"] = SchemaNode(
            **DEFAULT_PREDICT_KWARGS,
            needs={},
            uses=FlowsProvider,
            fn="provide_inference",
            config={},
            resource=Resource("flows_provider"),
        )
        predict_nodes["domain_provider"] = SchemaNode(
            **DEFAULT_PREDICT_KWARGS,
            needs={},
            uses=DomainProvider,
            fn="provide_inference",
            config={},
            resource=Resource("domain_provider"),
        )

        for idx, config in enumerate(predict_config["pipeline"]):
            component_name = config.pop("name")
            component = self._from_registry(component_name)
            component_name = f"{component_name}{idx}"
            if self.ComponentType.MODEL_LOADER in component.types:
                predict_nodes[f"provide_{component_name}"] = SchemaNode(
                    **DEFAULT_PREDICT_KWARGS,
                    needs={},
                    uses=component.clazz,
                    fn="provide",
                    config=config,
                )

            if component.types.intersection(
                {
                    self.ComponentType.MESSAGE_TOKENIZER,
                    self.ComponentType.MESSAGE_FEATURIZER,
                }
            ):
                last_run_node = self._add_nlu_predict_node_from_train(
                    predict_nodes,
                    component_name,
                    train_nodes,
                    last_run_node,
                    config,
                    from_resource=component.is_trainable,
                )
            elif component.types.intersection(
                {
                    self.ComponentType.INTENT_CLASSIFIER,
                    self.ComponentType.ENTITY_EXTRACTOR,
                    self.ComponentType.COMMAND_GENERATOR,
                    self.ComponentType.COEXISTENCE_ROUTER,
                }
            ):
                if component.is_trainable:
                    last_run_node = self._add_nlu_predict_node_from_train(
                        predict_nodes,
                        component_name,
                        train_nodes,
                        last_run_node,
                        config,
                        from_resource=component.is_trainable,
                    )
                else:
                    new_node = SchemaNode(
                        needs={"messages": last_run_node},
                        uses=component.clazz,
                        constructor_name="create",
                        fn="process",
                        config=config,
                    )

                    last_run_node = self._add_nlu_predict_node(
                        predict_nodes, new_node, component_name, last_run_node
                    )

        return last_run_node

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

        needs = self._get_needs_from_args(node.uses, "process")
        needs.update(self._get_model_provider_needs(predict_nodes, node.uses))
        needs["messages"] = last_run_node
        predict_nodes[node_name] = dataclasses.replace(
            node,
            needs=needs,
            fn="process",
            **DEFAULT_PREDICT_KWARGS,
        )

        return node_name

    def _add_core_predict_nodes(
        self,
        predict_config: Dict[Text, Any],
        predict_nodes: Dict[Text, SchemaNode],
        train_nodes: Dict[Text, SchemaNode],
        preprocessors: List[Text],
    ) -> None:
        predict_nodes["domain_provider"] = SchemaNode(
            **DEFAULT_PREDICT_KWARGS,
            needs={},
            uses=DomainProvider,
            fn="provide_inference",
            config={},
            resource=Resource("domain_provider"),
        )
        predict_nodes["story_graph_provider"] = SchemaNode(
            **DEFAULT_PREDICT_KWARGS,
            needs={},
            uses=StoryGraphProvider,
            fn="provide_inference",
            config={},
            resource=Resource("story_graph_provider"),
        )
        predict_nodes["flows_provider"] = SchemaNode(
            **DEFAULT_PREDICT_KWARGS,
            needs={},
            uses=FlowsProvider,
            fn="provide_inference",
            config={},
            resource=Resource("flows_provider"),
        )

        node_with_e2e_features = None

        if "end_to_end_features_provider" in train_nodes:
            node_with_e2e_features = self._add_end_to_end_features_for_inference(
                predict_nodes, preprocessors
            )

        predict_nodes["command_processor"] = SchemaNode(
            **DEFAULT_PREDICT_KWARGS,
            needs=self._get_needs_from_args(
                CommandProcessorComponent, "execute_commands"
            ),
            uses=CommandProcessorComponent,
            fn="execute_commands",
            config={},
            resource=Resource("command_processor"),
        )

        rule_policy_resource = None
        policies: List[Text] = []

        for idx, config in enumerate(predict_config["policies"]):
            component_name = config.pop("name")
            component = self._from_registry(component_name)

            train_node_name = f"train_{component_name}{idx}"
            node_name = f"run_{component_name}{idx}"

            from rasa.core.policies.rule_policy import RulePolicy

            if issubclass(component.clazz, RulePolicy) and not rule_policy_resource:
                rule_policy_resource = train_node_name

            needs = self._get_needs_from_args(
                train_nodes[train_node_name].uses, "predict_action_probabilities"
            )
            if (
                self.ComponentType.POLICY_WITH_END_TO_END_SUPPORT in component.types
                and node_with_e2e_features
            ):
                needs["precomputations"] = node_with_e2e_features

            predict_nodes[node_name] = dataclasses.replace(
                train_nodes[train_node_name],
                **DEFAULT_PREDICT_KWARGS,
                needs=needs,
                fn="predict_action_probabilities",
                resource=Resource(train_node_name),
            )
            policies.append(node_name)

        predict_nodes["rule_only_data_provider"] = SchemaNode(
            **DEFAULT_PREDICT_KWARGS,
            needs={},
            uses=RuleOnlyDataProvider,
            fn="provide",
            config={},
            resource=Resource(rule_policy_resource) if rule_policy_resource else None,
        )

        predict_nodes["select_prediction"] = SchemaNode(
            **DEFAULT_PREDICT_KWARGS,
            needs={
                **{f"policy{idx}": name for idx, name in enumerate(policies)},
                "domain": "domain_provider",
                "tracker": PLACEHOLDER_TRACKER,
            },
            uses=DefaultPolicyPredictionEnsemble,
            fn="combine_predictions_from_kwargs",
            config={},
        )

    def _add_end_to_end_features_for_inference(
        self, predict_nodes: Dict[Text, SchemaNode], preprocessors: List[Text]
    ) -> Text:
        predict_nodes["tracker_to_message_converter"] = SchemaNode(
            **DEFAULT_PREDICT_KWARGS,
            needs={"tracker": PLACEHOLDER_TRACKER},
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
            **DEFAULT_PREDICT_KWARGS,
            needs={"messages": last_node_name},
            uses=CoreFeaturizationCollector,
            fn="collect",
            config={},
        )
        return node_with_e2e_features

    @staticmethod
    def auto_configure(
        config_file_path: Optional[Text],
        config: Dict,
        training_type: Optional[TrainingType] = TrainingType.BOTH,
    ) -> Tuple[Dict[Text, Any], Set[str], Set[str]]:
        """Determine configuration from auto-filled configuration file.

        Keys that are provided and have a value in the file are kept. Keys that are not
        provided are configured automatically.

        Note that this needs to be called explicitly; ie. we cannot
        auto-configure automatically from importers because importers are not
        allowed to access code outside of `rasa.shared`.

        Args:
            config_file_path: The path to the configuration file.
            config: Configuration in dictionary format.
            training_type: Optional training type to auto-configure. By default
            both core and NLU will be auto-configured.
        """
        missing_keys = DefaultV1Recipe._get_missing_config_keys(config, training_type)
        keys_to_configure = DefaultV1Recipe._get_unspecified_autoconfigurable_keys(
            config, training_type
        )

        if keys_to_configure:
            config = DefaultV1Recipe.complete_config(config, keys_to_configure)
            DefaultV1Recipe._dump_config(
                config, config_file_path, missing_keys, keys_to_configure, training_type
            )

        return config, missing_keys, keys_to_configure

    @staticmethod
    def _get_unspecified_autoconfigurable_keys(
        config: Dict[Text, Any],
        training_type: Optional[TrainingType] = TrainingType.BOTH,
    ) -> Set[Text]:
        if training_type == TrainingType.NLU:
            all_keys = rasa.shared.constants.CONFIG_AUTOCONFIGURABLE_KEYS_NLU
        elif training_type == TrainingType.CORE:
            all_keys = rasa.shared.constants.CONFIG_AUTOCONFIGURABLE_KEYS_CORE
        else:
            all_keys = rasa.shared.constants.CONFIG_AUTOCONFIGURABLE_KEYS

        return {k for k in all_keys if config.get(k) is None}

    @staticmethod
    def _get_missing_config_keys(
        config: Dict[Text, Any],
        training_type: Optional[TrainingType] = TrainingType.BOTH,
    ) -> Set[Text]:
        if training_type == TrainingType.NLU:
            all_keys = rasa.shared.constants.CONFIG_KEYS_NLU
        elif training_type == TrainingType.CORE:
            all_keys = rasa.shared.constants.CONFIG_KEYS_CORE
        else:
            all_keys = rasa.shared.constants.CONFIG_KEYS

        return {k for k in all_keys if k not in config.keys()}

    @staticmethod
    def complete_config(
        config: Dict[Text, Any], keys_to_configure: Set[Text]
    ) -> Dict[Text, Any]:
        """Complete a config by adding automatic configuration for the specified keys.

        Args:
            config: The provided configuration.
            keys_to_configure: Keys to be configured automatically (e.g. `policies`).

        Returns:
            The resulting configuration including both the provided and
            the automatically configured keys.
        """
        import importlib_resources

        if keys_to_configure:
            logger.debug(
                f"The provided configuration does not contain the key(s) "
                f"{transform_collection_to_sentence(keys_to_configure)}. "
                f"Values will be provided from the default configuration."
            )

        default_config_file = str(
            importlib_resources.files(__name__)
            .joinpath("config_files")
            .joinpath("default_config.yml")
        )
        default_config = read_config_file(default_config_file)

        config = copy.deepcopy(config)
        for key in keys_to_configure:
            config[key] = default_config[key]

        return config

    @staticmethod
    def _dump_config(
        config: Dict[Text, Any],
        config_file_path: Text,
        missing_keys: Set[Text],
        auto_configured_keys: Set[Text],
        training_type: Optional[TrainingType] = TrainingType.BOTH,
    ) -> None:
        """Dump the automatically configured keys into the config file.

        The configuration provided in the file is kept as it is (preserving the order of
        keys and comments).
        For keys that were automatically configured, an explanatory
        comment is added and the automatically chosen configuration is
        added commented-out.
        If there are already blocks with comments from a previous auto
        configuration run, they are replaced with the new auto
        configuration.

        Args:
            config: The configuration including the automatically configured keys.
            config_file_path: The file into which the configuration should be dumped.
            missing_keys: Keys that need to be added to the config file.
            auto_configured_keys: Keys for which a commented out auto
            configuration section needs to be added to the config file.
            training_type: NLU, CORE or BOTH depending on which is trained.
        """
        config_as_expected = DefaultV1Recipe._is_config_file_as_expected(
            config_file_path, missing_keys, auto_configured_keys, training_type
        )
        if not config_as_expected:
            rasa.shared.utils.cli.print_error(
                f"The configuration file at '{config_file_path}' has been removed or "
                f"modified while the automatic configuration was running. The current "
                f"configuration will therefore not be dumped to the file. If you want "
                f"your model to use the configuration provided in "
                f"'{config_file_path}' you need to re-run training."
            )
            return

        DefaultV1Recipe._add_missing_config_keys_to_file(config_file_path, missing_keys)

        autoconfig_lines = DefaultV1Recipe._get_commented_out_autoconfig_lines(
            config, auto_configured_keys
        )

        current_config_content = rasa.shared.utils.io.read_file(config_file_path)
        current_config_lines = current_config_content.splitlines(keepends=True)

        updated_lines = DefaultV1Recipe._get_lines_including_autoconfig(
            current_config_lines, autoconfig_lines
        )

        rasa.shared.utils.io.write_text_file("".join(updated_lines), config_file_path)

        auto_configured_keys_text = transform_collection_to_sentence(
            auto_configured_keys
        )
        rasa.shared.utils.cli.print_info(
            f"The configuration for {auto_configured_keys_text} "
            f"was chosen automatically. "
            f"It was written into the config file at '{config_file_path}'."
        )

    @staticmethod
    def _is_config_file_as_expected(
        config_file_path: Text,
        missing_keys: Set[Text],
        auto_configured_keys: Set[Text],
        training_type: Optional[TrainingType] = TrainingType.BOTH,
    ) -> bool:
        try:
            content = read_config_file(config_file_path)
        except FileNotFoundException:
            content = {}

        return (
            bool(content)
            and missing_keys
            == DefaultV1Recipe._get_missing_config_keys(content, training_type)
            and auto_configured_keys
            == DefaultV1Recipe._get_unspecified_autoconfigurable_keys(
                content, training_type
            )
        )

    @staticmethod
    def _add_missing_config_keys_to_file(
        config_file_path: Text, missing_keys: Set[Text]
    ) -> None:
        if not missing_keys:
            return
        with open(
            config_file_path, "a", encoding=rasa.shared.utils.io.DEFAULT_ENCODING
        ) as f:
            for key in missing_keys:
                f.write(f"{key}:\n")

    @staticmethod
    def _get_lines_including_autoconfig(
        lines: List[Text], autoconfig_lines: Dict[Text, List[Text]]
    ) -> List[Text]:
        auto_configured_keys = autoconfig_lines.keys()

        lines_with_autoconfig = []
        remove_comments_until_next_uncommented_line = False
        for line in lines:
            insert_section = None

            # remove old auto configuration
            if remove_comments_until_next_uncommented_line:
                if line.startswith("#"):
                    continue
                remove_comments_until_next_uncommented_line = False

            # add an explanatory comment to autoconfigured sections
            for key in auto_configured_keys:
                if line.startswith(f"{key}:"):  # start of next auto-section
                    line = line + COMMENTS_FOR_KEYS[key]
                    insert_section = key
                    remove_comments_until_next_uncommented_line = True

            lines_with_autoconfig.append(line)

            if not insert_section:
                continue

            # add the autoconfiguration (commented out)
            lines_with_autoconfig += autoconfig_lines[insert_section]

        return lines_with_autoconfig

    @staticmethod
    def _get_commented_out_autoconfig_lines(
        config: Dict[Text, Any], auto_configured_keys: Set[Text]
    ) -> Dict[Text, List[Text]]:
        import ruamel.yaml
        import ruamel.yaml.compat

        yaml_parser = ruamel.yaml.YAML()
        yaml_parser.indent(mapping=2, sequence=4, offset=2)

        autoconfig_lines = {}

        for key in auto_configured_keys:
            stream = ruamel.yaml.compat.StringIO()
            yaml_parser.dump(config.get(key), stream)
            dump = stream.getvalue()

            lines = dump.split("\n")
            if not lines[-1]:
                lines = lines[:-1]  # yaml dump adds an empty line at the end
            lines = [f"# {line}\n" for line in lines]

            autoconfig_lines[key] = lines

        return autoconfig_lines
