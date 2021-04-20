from typing import Any, Dict, Optional, Text, Tuple, Type

from rasa.architecture_prototype.graph_components import TrainingDataReader
from rasa.architecture_prototype import graph
from rasa.nlu import registry
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
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


def train_and_process_component(
    component_class: Type[Component],
    input_task_name: Text,
    config: Optional[Dict[Text, Any]] = None,
    name: Optional[Text] = None,
    train_function: Text = "train",
    process_function: Text = "process_training_data",
    input_data_param_name: Text = "training_data",
) -> Tuple[Dict[Text, Any], Text]:
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
                "config": config,
                "needs": {
                    "resource_name": train_task_name,
                    input_data_param_name: input_task_name,
                },
            },
        },
        process_task_name,
    )


def train_component(
    component_class: Type[Component],
    input_task_name: Text,
    config: Optional[Dict[Text, Any]] = None,
    name: Optional[Text] = None,
    train_function: Text = "train",
    input_data_param_name: Text = "training_data",
) -> Tuple[Dict[Text, Any], Text]:
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
        input_task_name,
    )


def process_component(
    component_class: Type[Component],
    input_task_name: Text,
    config: Optional[Dict[Text, Any]] = None,
    name: Optional[Text] = None,
    process_function: Text = "process_training_data",
    input_data_param_name: Text = "training_data",
) -> Tuple[Dict[Text, Any], Text]:
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
                "persist": False,
            },
        },
        process_task_name,
    )


def nlu_config_to_train_graph(config: Text) -> Tuple[Dict[Text, Any], Text]:
    project = "examples/moodbot"

    config_dict = read_yaml(config)
    nlu_pipeline = config_dict["pipeline"]

    # TODO: get this information from the class?
    meta: Dict[Type[Component], Text] = {
        WhitespaceTokenizer: "process",
        RegexFeaturizer: "train_process",
        LexicalSyntacticFeaturizer: "train_process",
        CountVectorsFeaturizer: "train_process",
        DIETClassifier: "train",
        ResponseSelector: "train",
        EntitySynonymMapper: "train",
    }
    last_component_out = "load_data"
    nlu_train_graph = {
        "load_data": {
            "uses": TrainingDataReader,
            "fn": "read",
            "config": {"project": project},
            "needs": {},
            "persist": False,
        },
    }
    for i, component in enumerate(nlu_pipeline):
        component_name = component.pop("name")
        unique_component_name = f"{component_name}_{i}"
        component_class = registry.get_component_class(component_name)
        step_type = meta[component_class]
        config = {"component_config": component}
        builder = {
            "process": process_component,
            "train": train_component,
            "train_process": train_and_process_component,
        }[step_type]
        component_def, last_component_out = builder(
            component_class=component_class,
            input_task_name=last_component_out,
            name=unique_component_name,
            config=config,
        )
        nlu_train_graph.update(component_def)

    graph.fill_defaults(nlu_train_graph)

    return nlu_train_graph, last_component_out
