from typing import Text, Dict, Any, Type

import dask

from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.components import Component
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.nlu.training_data.formats import RasaYAMLReader
from rasa.shared.nlu.training_data.training_data import TrainingData


class RasaComponent:
    def __init__(
        self, component: Type[Component], config: Dict[Text, Any], fn_name: Text
    ) -> None:

        self._component = component(**config)
        self._run = getattr(component, fn_name)

    def __call__(self, *args: Any, **kwargs: Any) -> TrainingData:
        result = self._run(self._component, *args, **kwargs)

        return result


def create_graph():
    training_data = lambda f: RasaYAMLReader().read(f)
    tokenizer = RasaComponent(WhitespaceTokenizer, {}, "train")
    featurizer = RasaComponent(CountVectorsFeaturizer, {}, "train")

    classifier = RasaComponent(DIETClassifier, {}, "train")

    graph = {
        "load_data": (training_data, "examples/moodbot/data/nlu.yml"),
        "tokenize": (tokenizer, "load_data"),
        "train_featurizer": (featurizer, "tokenize"),
        "featurize": (
            RasaComponent(CountVectorsFeaturizer, {}, "process"),
            "train_featurizer",
            "tokenize",
        ),
        "classify": (classifier, "featurize"),
    }

    import dask.threaded

    dask.threaded.get(graph, "classify")
    # dask.visualize(graph, filename="graph.png")


class GraphTrainingData:
    def train(self):
        return RasaYAMLReader().read("examples/moodbot/data/nlu.yml")


def create_graph(name: Text, graph: Dict, registry: Dict[Text, Any]):
    # TODO: Catch cycles
    node_description = graph[name]
    params = []

    if name in registry:
        return registry[name]

    component = RasaComponent(
        node_description["uses"], {}, node_description.get("fn", "train")
    )

    # TODO: Do we need this?
    registry[name] = component

    for dep_name in node_description["needs"]:
        param = create_graph(dep_name, graph, registry)
        params.append(param)

    registry[name] = dask.delayed(component)(*params)

    return registry[name]


def test_create_graph_with_rasa_syntax():
    dsk = {
        "load_data": {"uses": TrainingData, "needs": []},
        "tokenize": {"uses": WhitespaceTokenizer, "needs": ["load_data"]},
        "train_featurizer": {"uses": CountVectorsFeaturizer, "needs": ["tokenize"]},
        "featurize": {
            "uses": CountVectorsFeaturizer,
            "fn": "process",
            "needs": ["train_featurizer", "tokenize"],
        },
        "classify": {"uses": DIETClassifier, "needs": ["featurize"]},
    }

    registry = {}

    graph = create_graph("classify", dsk, registry)

    graph.visualize("graph.png")
    graph.compute()


def test_graph():
    create_graph()


# TODO:
# 0. Run model: Persist current + load + predict
# 1. Try caching
# 2. Persistence
# 3.
