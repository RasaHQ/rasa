import dask

from rasa.architecture_prototype import graph
from rasa.architecture_prototype.graph import TrainingDataReader, RasaComponent
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

rasa_nlu_train_graph = {
    "load_data": {
        "uses": TrainingDataReader,
        "fn": "read",
        "config": {"filename": "examples/moodbot/data/nlu.yml"},
        "needs": [],
    },
    "tokenize": {
        "uses": WhitespaceTokenizer,
        "fn": "train",
        "config": {},
        "needs": ["load_data"],
    },
    "train_featurizer": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {},
        "needs": ["tokenize"],
    },
    "featurize": {
        "uses": CountVectorsFeaturizer,
        "fn": "process",
        "config": {},
        "needs": ["train_featurizer", "tokenize"],
    },
    "train_classifier": {
        "uses": DIETClassifier,
        "fn": "train",
        "config": {"component_config": {"epochs": 1}},
        "needs": ["featurize"],
    },
}


def test_create_graph_with_rasa_syntax():
    dask_graph = graph.convert_to_dask_graph(rasa_nlu_train_graph)

    assert dask_graph == {
        "load_data": (
            RasaComponent(
                TrainingDataReader,
                {"filename": "examples/moodbot/data/nlu.yml"},
                "read",
                "load_data",
            ),
        ),
        "tokenize": (
            RasaComponent(WhitespaceTokenizer, {}, "train", node_name="tokenize"),
            "load_data",
        ),
        "train_featurizer": (
            RasaComponent(
                CountVectorsFeaturizer, {}, "train", node_name="train_featurizer"
            ),
            "tokenize",
        ),
        "featurize": (
            RasaComponent(CountVectorsFeaturizer, {}, "process", node_name="featurize"),
            "train_featurizer",
            "tokenize",
        ),
        "train_classifier": (
            RasaComponent(
                DIETClassifier,
                {"component_config": {"epochs": 1}},
                fn_name="train",
                node_name="train_classifier",
            ),
            "featurize",
        ),
    }

    dask.visualize(dask_graph, filename="graph.png")
