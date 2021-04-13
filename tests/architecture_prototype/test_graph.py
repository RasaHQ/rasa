import dask

from rasa.architecture_prototype import graph
from rasa.architecture_prototype.graph import TrainingDataReader, RasaComponent
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
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

# We can omit `FallbackerClassifier` as this doesn't train
rasa_nlu_train_graph = {
    "load_data": {
        "uses": TrainingDataReader,
        "fn": "read",
        "config": {"filename": "examples/moodbot/data/nlu.yml"},
        "needs": {},
    },
    "tokenize": {
        "uses": WhitespaceTokenizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "load_data"},
    },
    # "train_regex_featurizer": {
    #     "uses": RegexFeaturizer,
    #     "fn": "train",
    #     "config": {},
    #     "needs": ["tokenize"],
    # },
    # "add_regex_features": {
    #     "uses": RegexFeaturizer,
    #     "fn": "process_training_data",
    #     "config": {},
    #     "needs": ["train_regex_featurizer", "tokenize"],
    # },
    # "train_lexical_featurizer": {
    #     "uses": LexicalSyntacticFeaturizer,
    #     "fn": "train",
    #     "config": {"component_config": {}},
    #     "needs": ["tokenize"],
    # },
    # "add_lexical_features": {
    #     "uses": LexicalSyntacticFeaturizer,
    #     "fn": "process_training_data",
    #     "config": {"component_config": {}},
    #     "needs": ["train_lexical_featurizer", "add_regex_features"],
    # },
    "train_count_featurizer1": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {"model_dir": "model", "filename": "count_featurizer1"},
        "needs": {"training_data": "tokenize"},
    },
    "add_count_features1": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "eager": False,
        "fn": "process_training_data",
        "config": {"model_dir": "model", "filename": "count_featurizer1"},
        "needs": {"training_data": "tokenize", "filename": "train_count_featurizer1"},
    },
    "train_count_featurizer2": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {"model_dir": "model", "filename": "count_featurizer2"},
        "needs": {"training_data": "tokenize"},
    },
    "add_count_features2": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "eager": False,
        "fn": "process_training_data",
        "config": {"model_dir": "model", "filename": "count_featurizer2"},
        "needs": {
            "filename": "train_count_featurizer2",
            "training_data": "add_count_features1",
        },
    },
    "train_classifier": {
        "uses": DIETClassifier,
        "fn": "train",
        "config": {"component_config": {"epochs": 1}},
        "needs": {"training_data": "add_count_features2"},
    },
    "train_response_selector": {
        "uses": ResponseSelector,
        "fn": "train",
        "config": {"component_config": {"epochs": 1}},
        "needs": {"training_data": "tokenize"},
    },
    "train_synonym_mapper": {
        "uses": EntitySynonymMapper,
        "config": {},
        "fn": "train",
        "needs": {"training_data": "tokenize"},
    },
}


def test_create_graph_with_rasa_syntax():
    dask_graph = graph.convert_to_dask_graph(rasa_nlu_train_graph)

    dask.visualize(dask_graph, filename="graph.png")


def test_train_nlu():
    trained_components = graph.run_as_dask_graph(
        rasa_nlu_train_graph,
        ["train_classifier", "train_response_selector", "train_synonym_mapper"],
    )

    assert isinstance(trained_components[0], DIETClassifier)
    assert isinstance(trained_components[1], ResponseSelector)
    assert isinstance(trained_components[2], EntitySynonymMapper)
