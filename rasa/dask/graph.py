from copy import deepcopy
from threading import Thread
from time import time
from typing import List, Text, Dict, Any, Type

import dask

from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.components import Component
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.nlu.training_data.formats import RasaYAMLReader
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData


class RasaComponent:
    def __init__(self, component: Any, config: Dict[Text, Any], fn_name: Text) -> None:

        self._component = component(**config)
        self._run = getattr(component, fn_name)

    def __call__(self, *args: Any, **kwargs: Any) -> TrainingData:
        result = self._run(self._component, *args, **kwargs)

        return result


class ModelPersistor:
    def __init__(self, path):
        self.path = path

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


def test_train_graph():

    classifier = RasaComponent(
        DIETClassifier, {"component_config": {"epochs": 100}}, "train"
    )

    graph = {
        "load_data": (
            lambda f: RasaYAMLReader().read(f),
            "examples/moodbot/data/nlu.yml",
        ),
        "tokenize": (RasaComponent(WhitespaceTokenizer, {}, "train"), "load_data"),
        "train_featurizer": (
            RasaComponent(CountVectorsFeaturizer, {}, "train"),
            "tokenize",
        ),
        "featurize": (
            RasaComponent(CountVectorsFeaturizer, {"for_training": True}, "process"),
            "train_featurizer", "tokenize"
        ),
        "persist": (
            RasaComponent(ModelPersistor, {"path": "model"}, "save"),
            "train_featurizer", "train_classifier",
        ),
        "train_classifier": (classifier, "featurize"),
    }

    import dask.threaded

    dask.threaded.get(graph, "persist")
    # dask.visualize(graph, filename="graph.png")


def test_run_graph():
    graph = {
        "get_message": (lambda x: deepcopy(x), Message.build("hello")),
        "load_featurizer": (
            RasaComponent(
                ModelLoader,
                {"component_class": CountVectorsFeaturizer, "path": "model"},
                "load",
            ),
        ),
        "load_classifier": (
            RasaComponent(
                ModelLoader,
                {"component_class": DIETClassifier, "path": "model"},
                "load",
            ),
        ),
        "tokenize": (RasaComponent(WhitespaceTokenizer, {}, "process"), "get_message"),
        "featurize": (
            RasaComponent(CountVectorsFeaturizer, {}, "process"),
            "load_featurizer", "tokenize",
        ),
        "classify": (
            RasaComponent(DIETClassifier, {}, "process"),
            "load_classifier", "featurize",
        ),
    }

    import dask.threaded

    t = time()


    def run_the_ting():
        result = dask.threaded.get(graph, "classify")
        print(result.as_dict()['intent'])

    threads = []
    for i in range(100):
        threads.append(Thread(target=run_the_ting))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print(time() - t)
    # dask.visualize(graph, filename="graph.png")


# class GraphTrainingData:
#     def train(self):
#         return RasaYAMLReader().read("examples/moodbot/data/nlu.yml")

#
# def create_graph(name: Text, graph: Dict, registry: Dict[Text, Any]):
#     # TODO: Catch cycles
#     node_description = graph[name]
#     params = []
#
#     if name in registry:
#         return registry[name]
#
#     component = RasaComponent(
#         node_description["uses"], {}, node_description.get("fn", "train")
#     )
#
#     # TODO: Do we need this?
#     registry[name] = component
#
#     for dep_name in node_description["needs"]:
#         param = create_graph(dep_name, graph, registry)
#         params.append(param)
#
#     registry[name] = dask.delayed(component)(*params)
#
#     return registry[name]
#

# def test_create_graph_with_rasa_syntax():
#     dsk = {
#         "load_data": {"uses": TrainingData, "needs": []},
#         "tokenize": {"uses": WhitespaceTokenizer, "needs": ["load_data"]},
#         "train_featurizer": {"uses": CountVectorsFeaturizer, "needs": ["tokenize"]},
#         "featurize": {
#             "uses": CountVectorsFeaturizer,
#             "fn": "process",
#             "needs": ["train_featurizer", "tokenize"],
#         },
#         "classify": {"uses": DIETClassifier, "needs": ["featurize"]},
#     }
#
#     registry = {}
#
#     graph = create_graph("classify", dsk, registry)
#
#     graph.visualize("graph.png")
#     graph.compute()
#


# TODO:
# 0. Run model: Persist current + load + predict
# 1. Try caching
# 2. Persistence
# 3.
