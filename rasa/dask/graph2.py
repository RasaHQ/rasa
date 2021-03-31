import hashlib
import inspect
import json
import pickle
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from threading import Thread
from time import time
from typing import List, Text, Dict, Any, Type, Optional

import dask
import dask.threaded
import cloudpickle

from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.components import Component
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.nlu.training_data.formats import RasaYAMLReader
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils.common import class_from_module_path

fingerprint_cache = {}

fingerprint_graph_results = {}

def stable_hash(data):
    return hashlib.sha256(data.encode()).hexdigest()


class Fingerprint:
    def __init__(self, val: str):
        self._val = val

    def __eq__(self, other):
        if not isinstance(other, Fingerprint):
            return NotImplemented
        else:
            return self._val == other._val


class RasaComponent:
    def __init__(
        self,
        component: Any,
        config: Dict[Text, Any],
        fn_name: Text,
        node_name: Text,
        fingermode = False,
    ) -> None:
        self._component = component(**config)
        self._config = config
        self._run = getattr(component, fn_name)
        self._node_name = node_name
        self._fingermode = fingermode

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._fingermode:
            return self.finger_run(*args, **kwargs)
        else:
            return self.run(*args, **kwargs)

    def finger_run(self, *args, **kwargs):
        fingerprint_key = self.get_fingerprint(args)
        cached = fingerprint_cache.get(fingerprint_key)
        if cached:
            fingerprint_graph_results[self._node_name] = 'noop'
            return cached
        else:
            fingerprint_graph_results[self._node_name] = 'run'
            for dependent in train_graph_schema[self._node_name][1]:
                if fingerprint_graph_results[dependent] == 'noop':
                    fingerprint_graph_results[dependent] = 'use_cache'
            return ""

    def run(self, *args: Any, **kwargs: Any) -> Any:
        fingerprint_key = self.get_fingerprint(args)
        result = self._run(self._component, *args, **kwargs)

        if result:
            result_fingerprint = stable_hash(result.fingerprint())
            fingerprint_cache[fingerprint_key] = Fingerprint(result_fingerprint)
        else:
            fingerprint_cache[fingerprint_key] = Fingerprint('True')

        return result

    def get_fingerprint(self, args):
        arg_fingerprints = []
        for arg in args:
            if isinstance(arg, str):
                arg_fingerprints.append(stable_hash(arg))
            elif isinstance(arg, Fingerprint):
                arg_fingerprints.append(arg._val)
            else:
                arg_fingerprints.append(stable_hash(arg.fingerprint()))

        args_fingerprint = "_".join(arg_fingerprints)
        config_fingerprint = stable_hash(json.dumps(self._config))
        fingerprint = self._node_name + "_" + config_fingerprint + '_' + args_fingerprint
        return fingerprint


class ModelPersistor:
    def __init__(self, path):
        self.path = path
        Path(path).mkdir(exist_ok=True)

    def save(self, *components: Component):
        for component in components:
            component.persist(component.name, self.path)


class ModelLoader:
    def __init__(self, component_class: Type[Component], path):
        self.loaded_component = component_class.load(
            {"file": component_class.name,}, path
        )

    def load(self):
        return self.loaded_component


class DataReader:
    def read(self, f):
        return RasaYAMLReader().read(f)


train_graph_schema = {
    "load_data": (
        (DataReader, 'read', {}),
        ["examples/moodbot/data/nlu.yml"],
    ),
    "tokenize": (
        (WhitespaceTokenizer, "train", {}),
        ("load_data",)
    ),
    "train_featurizer": (
        (CountVectorsFeaturizer, "train", {}),
        ("tokenize",)
    ),
    "featurize": (
        (CountVectorsFeaturizer, "process", {"for_training": True}),
        ("train_featurizer", "tokenize",)
    ),
    "persist_classifier": (
        (ModelPersistor, "save", {"path": "model"}),
        ("train_classifier",)
    ),
    "persist_featurizer": (
        (ModelPersistor, "save", {"path": "model"}),
        ("train_featurizer",)
    ),
    "train_classifier": (
        (DIETClassifier, "train", {"component_config": {"epochs": 1}}),
        ("featurize",)
    ),
}


class CachedComponent:
    def __init__(self, *args, **kwargs):
        pass

    def from_cache(self):
        return None

def create_train_graph(graph_schema, finger_results=None):
    finger_results = finger_results if finger_results else {}
    graph = {}
    for node_name, definition in graph_schema.items():
        task_definition, dependents = definition
        component, function_name, config = task_definition

        finger_result = finger_results.get(node_name, 'run')
        if finger_result == 'noop':
            continue
        elif finger_result == 'use_cache':
            graph[node_name] = (
                CachedComponent,
                *dependents
            )
        else:
            graph[node_name] = (
                RasaComponent(component, config, function_name, node_name),
                *dependents
            )
    return graph


def create_finger_train_graph(graph_schema):
    graph = {}
    for node_name, definition in graph_schema.items():
        task_definition, dependents = definition
        component, function_name, config = task_definition
        graph[node_name] = (
            RasaComponent(component, config, function_name, node_name, fingermode=True),
            *dependents
        )
    return graph


def test_train_graph():
    first_train_graph = create_train_graph(train_graph_schema)
    dask.visualize(first_train_graph, filename="first_train_graph.png")
    dask.threaded.get(first_train_graph, ["persist_featurizer", "persist_classifier"])

    finger_graph = create_finger_train_graph(train_graph_schema)
    train_graph_schema['train_classifier'][0][2]['component_config']['epochs'] = 2
    dask.threaded.get(finger_graph, ["persist_featurizer", "persist_classifier"])
    pprint(fingerprint_graph_results)
    second_train_graph = create_train_graph(train_graph_schema, fingerprint_graph_results)
    dask.visualize(second_train_graph, filename="second_train_graph.png")


