from copy import deepcopy
import inspect
from typing import Any, Dict, List, Optional, Text, Tuple, Type

from rasa.architecture_prototype.graph import GraphSchema
from rasa.architecture_prototype.graph_components import (
    DomainReader,
    MessageCreator,
    MessageToE2EFeatureConverter,
    NLUMessageConverter,
    NLUPredictionToHistoryAdder,
    ProjectProvider,
    StoryGraphReader,
    StoryToTrainingDataConverter,
    TrackerGenerator,
    TrackerLoader,
    TrainingDataReader,
)
from rasa.core.channels import UserMessage
from rasa.core.policies import SimplePolicyEnsemble
from rasa.nlu import registry
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.classifiers.fallback_classifier import FallbackClassifier
from rasa.nlu.components import Component
from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.nlu.selectors.response_selector import ResponseSelector
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.utils.io import read_yaml
import rasa.core.registry

# TODO: cleanup


def _train_and_process_component(
    component_class: Type[Component],
    input_task_name: Text,
    config: Optional[Dict[Text, Any]] = None,
    name: Optional[Text] = None,
    train_function: Text = "train",
    process_function: Text = "process_training_data",
    input_data_param_name: Text = "training_data",
) -> Tuple[Dict[Text, Any], List[Text], Text]:
    name = name if name else component_class.name
    train_task_name = f"train_{name}"
    process_task_name = f"process_{name}"
    config = config if config else {}
    return (
        {
            train_task_name: {
                "uses": component_class,
                "fn": train_function,
                "config": config,
                "needs": {input_data_param_name: input_task_name},
            },
            process_task_name: {
                "uses": component_class,
                "fn": process_function,
                "constructor_name": "load",
                "eager": False,
                "config": config,
                "needs": {
                    "resource_name": train_task_name,
                    input_data_param_name: input_task_name,
                },
            },
        },
        [train_task_name, process_task_name],
        process_task_name,
    )


def _train_component(
    component_class: Type[Component],
    input_task_name: Text,
    config: Optional[Dict[Text, Any]] = None,
    name: Optional[Text] = None,
    train_function: Text = "train",
    input_data_param_name: Text = "training_data",
) -> Tuple[Dict[Text, Any], List[Text], Text]:
    name = name if name else component_class.name
    train_task_name = f"train_{name}"
    config = config if config else {}
    return (
        {
            train_task_name: {
                "uses": component_class,
                "fn": train_function,
                "config": config,
                "needs": {input_data_param_name: input_task_name},
            },
        },
        [train_task_name],
        input_task_name,
    )


def _process_component(
    component_class: Type[Component],
    input_task_name: Text,
    config: Optional[Dict[Text, Any]] = None,
    name: Optional[Text] = None,
    process_function: Text = "process_training_data",
    input_data_param_name: Text = "training_data",
) -> Tuple[Dict[Text, Any], List[Text], Text]:
    name = name if name else component_class.name
    process_task_name = f"process_{name}"
    config = config if config else {}
    return (
        {
            process_task_name: {
                "uses": component_class,
                "fn": process_function,
                "config": config,
                "needs": {input_data_param_name: input_task_name,},
                "persistor": False,
            },
        },
        [process_task_name],
        process_task_name,
    )


def _nlu_config_to_train_graph_schema(
    config: Dict[Text, Any],
    component_namespace: Optional[Text] = None,
    input_task: Optional[Text] = None,
    only_process: bool = False,
) -> Tuple[Dict[Text, Any], List[Text]]:
    nlu_pipeline = deepcopy(config["pipeline"])
    meta: Dict[Type[Component], Text] = {
        WhitespaceTokenizer: "process",
        RegexFeaturizer: "train_process",
        LexicalSyntacticFeaturizer: "train_process",
        CountVectorsFeaturizer: "train_process",
        DIETClassifier: "train",
        ResponseSelector: "train",
        EntitySynonymMapper: "train",
        FallbackClassifier: None,
    }
    if input_task:
        last_component_out = input_task
        nlu_train_graph = {}
    else:
        last_component_out = "load_data"
        nlu_train_graph = {
            "load_data": {
                "uses": TrainingDataReader,
                "fn": "read",
                "config": {},
                "needs": {"project": "get_project"},
                "persistor": False,
            },
        }

    train_outputs = []
    for i, component in enumerate(nlu_pipeline):
        component_name = component.pop("name")
        unique_component_name = f"{component_name}_{i}"
        if component_namespace:
            unique_component_name = f"{component_namespace}_{unique_component_name}"
        component_class = registry.get_component_class(component_name)
        step_type = meta[component_class]
        if step_type == "train" and only_process:
            continue
        config = component
        builder = {
            "process": _process_component,
            "train": _train_component,
            "train_process": _train_and_process_component,
        }.get(step_type)
        if not builder:
            continue
        component_def, task_names, last_component_out = builder(
            component_class=component_class,
            input_task_name=last_component_out,
            name=unique_component_name,
            config=config,
        )
        nlu_train_graph.update(component_def)
        if step_type == "train":
            train_outputs.append(*task_names)

    return nlu_train_graph, train_outputs if train_outputs else [last_component_out]


def _nlu_config_to_predict_graph_schema(
    config: Dict[Text, Any],
    input_task: Optional[Text],
    component_namespace: Optional[Text] = None,
    classify: bool = True,
) -> Tuple[Dict[Text, Any], Text]:
    nlu_pipeline = deepcopy(config["pipeline"])
    meta: Dict[Type[Component], Text] = {
        WhitespaceTokenizer: "process",
        RegexFeaturizer: "process",
        LexicalSyntacticFeaturizer: "process",
        CountVectorsFeaturizer: "process",
        DIETClassifier: "classify",
        ResponseSelector: "classify",
        EntitySynonymMapper: "classify",
        FallbackClassifier: "classify",
    }
    last_component_out = input_task
    nlu_predict_graph = {}
    for i, component in enumerate(nlu_pipeline):
        component_name = component.pop("name")
        unique_component_name = f"{component_name}_{i}"
        if component_namespace:
            unique_component_name = f"{component_namespace}_{unique_component_name}"
        resource_name = f"train_{unique_component_name}"
        component_class = registry.get_component_class(component_name)
        step_type = meta[component_class]
        if step_type == "classify" and not classify:
            continue
        config = component
        component_def = {
            unique_component_name: {
                "uses": component_class,
                "constructor_name": "load",
                "fn": "process",
                "config": {"resource_name": resource_name, **config},
                "needs": {"messages": last_component_out,},
            },
        }
        nlu_predict_graph.update(component_def)
        last_component_out = unique_component_name

    return nlu_predict_graph, last_component_out


def _core_config_to_train_graph_schema(
    config: Dict[Text, Any]
) -> Tuple[Dict[Text, Any], List[Text]]:
    policies = deepcopy(config["policies"])
    core_train_graph = {
        "load_domain": {
            "uses": DomainReader,
            "fn": "read",
            "config": {},
            "needs": {"project": "get_project"},
        },
        "load_stories": {
            "uses": StoryGraphReader,
            "fn": "read",
            "config": {},
            "needs": {"project": "get_project"},
            "persistor": False,
        },
        "generate_trackers": {
            "uses": TrackerGenerator,
            "fn": "generate",
            "config": {},
            "needs": {"domain": "load_domain", "story_graph": "load_stories"},
            "persistor": False,
        },
    }
    policy_names = []
    e2e = False
    for i, policy in enumerate(policies):
        policy_name = policy.pop("name")
        unique_policy_name = f"{policy_name}_{i}"
        policy_names.append(unique_policy_name)
        policy_class = rasa.core.registry.policy_from_module_path(policy_name)
        policy_step = {
            unique_policy_name: {
                "uses": policy_class,
                "fn": "train",
                "config": policy,
                "needs": {
                    "training_trackers": "generate_trackers",
                    "domain": "load_domain",
                },
            },
        }
        if "e2e_features" in inspect.signature(policy_class.train).parameters:
            policy_step[unique_policy_name]["needs"][
                "e2e_features"
            ] = "create_e2e_lookup"
            e2e = True
        core_train_graph.update(policy_step)

    if e2e:
        core_train_graph["convert_stories_for_nlu"] = {
            "uses": StoryToTrainingDataConverter,
            "fn": "convert_for_training",
            "config": {},
            "needs": {"story_graph": "load_stories", "domain": "load_domain"},
            "persistor": False,
        }
        nlu_train_graph_schema, nlu_outs = _nlu_config_to_train_graph_schema(
            config,
            component_namespace="core",
            input_task="convert_stories_for_nlu",
            only_process=True,
        )
        core_train_graph.update(nlu_train_graph_schema)
        core_train_graph["create_e2e_lookup"] = {
            "uses": MessageToE2EFeatureConverter,
            "fn": "convert",
            "config": {},
            "needs": {"messages": nlu_outs[0]},
            "persistor": False,
        }

    return core_train_graph, policy_names


def _core_config_to_predict_graph_schema(
    config: Dict[Text, Any]
) -> Tuple[Dict[Text, Any], List[Text]]:
    core_predict_graph = {}
    policies = deepcopy(config["policies"])
    policy_names = []
    e2e = False
    for i, policy in enumerate(policies):
        policy_name = policy.pop("name")
        unique_policy_name = f"{policy_name}_{i}"
        policy_names.append(unique_policy_name)
        policy_class = rasa.core.registry.policy_from_module_path(policy_name)
        policy_step = {
            unique_policy_name: {
                "uses": policy_class,
                "constructor_name": "load",
                "fn": "predict_action_probabilities",
                "config": {"resource_name": unique_policy_name, **policy},
                "needs": {"tracker": "add_parsed_nlu_message", "domain": "load_domain"},
            },
        }
        if (
            "e2e_features"
            in inspect.signature(policy_class.predict_action_probabilities).parameters
        ):
            policy_step[unique_policy_name]["needs"][
                "e2e_features"
            ] = "create_e2e_lookup"
            e2e = True
        core_predict_graph.update(policy_step)

    if e2e:
        nlu_e2e_predict_graph_schema, nlu_e2e_out = _nlu_config_to_predict_graph_schema(
            config,
            input_task="convert_tracker_for_e2e",
            classify=False,
            component_namespace="core",
        )
        e2e_part = {
            "convert_tracker_for_e2e": {
                "uses": StoryToTrainingDataConverter,
                "fn": "convert_for_inference",
                "config": {},
                "needs": {"tracker": "add_parsed_nlu_message",},
                "persistor": False,
            },
            "create_e2e_lookup": {
                "uses": MessageToE2EFeatureConverter,
                "fn": "convert",
                "config": {},
                "needs": {"messages": nlu_e2e_out},
                "persistor": False,
            },
            **nlu_e2e_predict_graph_schema,
        }
        core_predict_graph.update(e2e_part)

    return core_predict_graph, policy_names


def config_to_train_graph_schema(config: Text) -> GraphSchema:
    """Create a training graph schema from the current rasa config."""
    config_dict = read_yaml(config)
    nlu_train_graph_schema, nlu_outs = _nlu_config_to_train_graph_schema(config_dict)
    core_train_graph_schema, core_outs = _core_config_to_train_graph_schema(config_dict)
    return {
        "get_project": {
            "uses": ProjectProvider,
            "fn": "get",
            "config": {"project": None},
            "needs": {},
            "persistor": False,
        },
        **core_train_graph_schema,
        **nlu_train_graph_schema,
        "targets": [*core_outs, *nlu_outs],
    }


def config_to_predict_graph_schema(config: Text,) -> GraphSchema:
    """Create a prediction graph schema from the current rasa config."""
    config_dict = read_yaml(config)

    nlu_predict_graph_schema, nlu_out = _nlu_config_to_predict_graph_schema(
        config_dict, input_task="convert_message_to_nlu", classify=True,
    )
    core_predict_graph_schema, core_outs = _core_config_to_predict_graph_schema(
        config_dict
    )

    predict_graph = {
        "load_user_message": {
            "uses": MessageCreator,
            "fn": "create",
            "config": {"message": None},
            "needs": {},
            "persistor": False,
        },
        "convert_message_to_nlu": {
            "uses": NLUMessageConverter,
            "fn": "convert",
            "config": {},
            "needs": {"message": "load_user_message"},
            "persistor": False,
        },
        "load_history": {
            "uses": TrackerLoader,
            "fn": "load",
            "needs": {},
            "config": {"tracker": None,},
            "persistor": False,
        },
        "add_parsed_nlu_message": {
            "uses": NLUPredictionToHistoryAdder,
            "fn": "merge",
            "needs": {
                "tracker": "load_history",
                "initial_user_message": "load_user_message",
                "parsed_messages": nlu_out,
                "domain": "load_domain",
            },
            "config": {},
            "persistor": False,
        },
        "load_domain": {
            "uses": DomainReader,
            "constructor_name": "load",
            "fn": "provide",
            "config": {"resource_name": "load_domain"},
            "needs": {},
        },
        "select_prediction": {
            "uses": SimplePolicyEnsemble,
            "fn": "probabilities_using_best_policy",
            "config": {},
            "persistor": False,
            "needs": {
                "tracker": "add_parsed_nlu_message",
                "domain": "load_domain",
                **{f"{p}_prediction": p for p in core_outs},
            },
        },
    }

    return {
        **predict_graph,
        **nlu_predict_graph_schema,
        **core_predict_graph_schema,
        "targets": ["select_prediction"],
    }
