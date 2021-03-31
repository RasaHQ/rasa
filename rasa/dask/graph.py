import inspect
import pickle
from copy import deepcopy
from pathlib import Path
from threading import Thread
from time import time
from typing import List, Text, Dict, Any, Type, Optional

import dask
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


class Cache:
    def __init__(self, path: Path):
        self.path = path

    def _cache_path(self, key: Text, inputs: List[Any]) -> Path:
        fingerprints = [str(hash(input_data)) for input_data in inputs]

        return self.path / f"{key}_{'_'.join(fingerprints)}"

    def put(self, key: Text, inputs: List[Any], output: Any):
        if output is None:
            return

        cache_path = self._cache_path(key, inputs)
        module = f"{inspect.getmodule(output).__name__}.{output.__class__.__name__}"

        module_file = cache_path.with_suffix(".module")
        module_file.write_text(module)

        cached_file = cache_path.with_suffix(".cached")
        cached_file.write_bytes(output.serialize())

    def get(self, key: Text, inputs: List[Any]) -> Optional[Any]:
        cache_path = self._cache_path(key, inputs)
        module_file = cache_path.with_suffix(".module")

        if not module_file.exists():
            return None

        module = module_file.read_text()
        cached = cache_path.with_suffix(".cached").read_bytes()

        cls = class_from_module_path(module)
        cls.deserialize(cached)


path = Path(".cache")
path.mkdir(exist_ok=True)

cache = Cache(path)


class RasaComponent:
    def __init__(
        self,
        component: Any,
        config: Dict[Text, Any],
        fn_name: Text,
        node_name: Text,
        cache: "Cache" = cache,
    ) -> None:
        self._cache = cache
        self._component = component(**config)
        self._run = getattr(component, fn_name)
        self._cache_key = node_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        cached = self._cache.get(self._cache_key, args)
        if cached:

            return cached

        result = self._run(self._component, *args, **kwargs)

        self._cache.put(self._cache_key, args, result)

        return result


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


def test_train_graph():
    graph = {
        "load_data": (
            lambda f: RasaYAMLReader().read(f),
            "examples/moodbot/data/nlu.yml",
        ),
        "tokenize": (
            RasaComponent(WhitespaceTokenizer, {}, "train", "tokenize"),
            "load_data",
        ),
        "train_featurizer": (
            RasaComponent(CountVectorsFeaturizer, {}, "train", "treain_featurizer"),
            "tokenize",
        ),
        "featurize": (
            RasaComponent(
                CountVectorsFeaturizer, {"for_training": True}, "process", "featurize"
            ),
            "train_featurizer",
            "tokenize",
        ),
        "persist": (
            RasaComponent(ModelPersistor, {"path": "model"}, "save", "model_persistor"),
            "train_featurizer",
            "train_classifier",
        ),
        "train_classifier": (
            RasaComponent(
                DIETClassifier,
                {"component_config": {"epochs": 100}},
                fn_name="train",
                node_name="train_classifier",
            ),
            "featurize",
        ),
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
            "load_featurizer",
            "tokenize",
        ),
        "classify": (
            RasaComponent(DIETClassifier, {}, "process"),
            "load_classifier",
            "featurize",
        ),
    }

    import dask.threaded

    t = time()

    def run_the_ting():
        result = dask.threaded.get(graph, "classify")
        print(result.as_dict()["intent"])

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
