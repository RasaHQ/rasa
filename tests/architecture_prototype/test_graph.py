import dask

from rasa.architecture_prototype import graph
from rasa.architecture_prototype.graph import (
    DomainReader,
    GeneratedStoryReader,
    TrainingDataReader,
    StoryToTrainingDataConverter,
    StoryGraphReader,
    MessageToE2EFeatureConverter,
)
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.policies.ted_policy import TEDPolicy
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
        "config": {"project": "examples/moodbot"},
        "needs": {},
    },
    "tokenize": {
        "uses": WhitespaceTokenizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "load_data"},
    },
    "train_regex_featurizer": {
        "uses": RegexFeaturizer,
        "fn": "train",
        "config": {
            "component_config": {"model_dir": "model", "filename": "regex_featurizer"}
        },
        "needs": {"training_data": "tokenize"},
    },
    "add_regex_features": {
        "uses": RegexFeaturizer,
        "fn": "process_training_data",
        "config": {},
        "needs": {"filename": "train_regex_featurizer", "training_data": "tokenize",},
    },
    "train_lexical_featurizer": {
        "uses": LexicalSyntacticFeaturizer,
        "fn": "train",
        "config": {
            "component_config": {"model_dir": "model", "filename": "lexical_featurizer"}
        },
        "needs": {"training_data": "tokenize"},
    },
    "add_lexical_features": {
        "uses": LexicalSyntacticFeaturizer,
        "fn": "process_training_data",
        "config": {
            "component_config": {"model_dir": "model", "filename": "lexical_featurizer"}
        },
        "needs": {
            "training_data": "add_regex_features",
            "filename": "train_lexical_featurizer",
        },
    },
    "train_count_featurizer1": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {
            "component_config": {"model_dir": "model", "filename": "count_featurizer1"}
        },
        "needs": {"training_data": "tokenize"},
    },
    "add_count_features1": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "eager": False,
        "fn": "process_training_data",
        "config": {
            "component_config": {"model_dir": "model", "filename": "count_featurizer1"}
        },
        "needs": {
            "training_data": "add_lexical_features",
            "filename": "train_count_featurizer1",
        },
    },
    "train_count_featurizer2": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {
            "component_config": {"model_dir": "model", "filename": "count_featurizer2"}
        },
        "needs": {"training_data": "tokenize"},
    },
    "add_count_features2": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "eager": False,
        "fn": "process_training_data",
        "config": {
            "component_config": {"model_dir": "model", "filename": "count_featurizer2"}
        },
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
        "needs": {"training_data": "add_count_features2"},
    },
    "train_synonym_mapper": {
        "uses": EntitySynonymMapper,
        "config": {},
        "fn": "train",
        "needs": {"training_data": "add_count_features2"},
    },
}


def test_create_graph_with_rasa_syntax():
    dask_graph = graph.convert_to_dask_graph(rasa_nlu_train_graph)

    dask.visualize(dask_graph, filename="graph.png")


def test_train_nlu():
    graph.run_as_dask_graph(
        rasa_nlu_train_graph,
        ["train_classifier", "train_response_selector", "train_synonym_mapper"],
    )


full_model_train_graph = {
    "load_domain": {
        "uses": DomainReader,
        "fn": "read",
        "config": {"project": "examples/moodbot"},
        "needs": {},
    },
    "load_stories": {
        "uses": GeneratedStoryReader,
        "fn": "read",
        "config": {"project": "examples/moodbot"},
        "needs": {"domain": "load_domain"},
    },
    "load_stories_simple": {
        "uses": StoryGraphReader,
        "fn": "read",
        "config": {"project": "examples/moodbot"},
        "needs": {"domain": "load_domain",},
    },
    "convert_stories_for_nlu": {
        "uses": StoryToTrainingDataConverter,
        "fn": "convert",
        "config": {},
        "needs": {"story_graph": "load_stories_simple"},
    },
    "tokenize": {
        "uses": WhitespaceTokenizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "convert_stories_for_nlu"},
    },
    "train_regex_featurizer": {
        "uses": RegexFeaturizer,
        "fn": "train",
        "config": {
            "component_config": {"model_dir": "model", "filename": "regex_featurizer"}
        },
        "needs": {"training_data": "tokenize"},
    },
    "add_regex_features": {
        "uses": RegexFeaturizer,
        "fn": "process_training_data",
        "config": {},
        "needs": {"filename": "train_regex_featurizer", "training_data": "tokenize",},
    },
    "train_lexical_featurizer": {
        "uses": LexicalSyntacticFeaturizer,
        "fn": "train",
        "config": {
            "component_config": {"model_dir": "model", "filename": "lexical_featurizer"}
        },
        "needs": {"training_data": "tokenize"},
    },
    "add_lexical_features": {
        "uses": LexicalSyntacticFeaturizer,
        "fn": "process_training_data",
        "config": {
            "component_config": {"model_dir": "model", "filename": "lexical_featurizer"}
        },
        "needs": {
            "training_data": "add_regex_features",
            "filename": "train_lexical_featurizer",
        },
    },
    "train_count_featurizer1": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {
            "component_config": {"model_dir": "model", "filename": "count_featurizer1"}
        },
        "needs": {"training_data": "tokenize"},
    },
    "add_count_features1": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "eager": False,
        "fn": "process_training_data",
        "config": {
            "component_config": {"model_dir": "model", "filename": "count_featurizer1"}
        },
        "needs": {
            "training_data": "add_lexical_features",
            "filename": "train_count_featurizer1",
        },
    },
    "train_count_featurizer2": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {
            "component_config": {"model_dir": "model", "filename": "count_featurizer2"}
        },
        "needs": {"training_data": "tokenize"},
    },
    "add_count_features2": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "eager": False,
        "fn": "process_training_data",
        "config": {
            "component_config": {"model_dir": "model", "filename": "count_featurizer2"}
        },
        "needs": {
            "filename": "train_count_featurizer2",
            "training_data": "add_count_features1",
        },
    },
    "merge_nlu_features": {
        "uses": MessageToE2EFeatureConverter,
        "fn": "convert",
        "config": {},
        "needs": {"training_data": "add_count_features2",},
    },
    "train_memoization_policy": {
        "uses": MemoizationPolicy,
        "fn": "train",
        "config": {},
        "needs": {"training_trackers": "load_stories", "domain": "load_domain"},
    },
    "train_rule_policy": {
        "uses": RulePolicy,
        "fn": "train",
        "config": {},
        "needs": {"training_trackers": "load_stories", "domain": "load_domain"},
    },
    "train_ted_policy": {
        "uses": TEDPolicy,
        "fn": "train",
        "config": {},
        "needs": {
            "e2e_features": "merge_nlu_features",
            "training_trackers": "load_stories",
            "domain": "load_domain",
        },
    },
}


def test_train_full_model():
    trained_components = graph.run_as_dask_graph(
        full_model_train_graph,
        ["train_memoization_policy", "train_ted_policy", "train_rule_policy"],
    )

    print(trained_components)
