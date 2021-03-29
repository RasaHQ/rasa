from typing import Text, Dict, Any, Type

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

    import dask.multiprocessing

    dask.multiprocessing.get(graph, "classify")
    # dask.visualize(graph, filename="graph.png")


def test_graph():
    create_graph()


# TODO:
# 0. Run model: Persist current + load + predict
# 1. Try caching
# 2. Persistence
# 3.
