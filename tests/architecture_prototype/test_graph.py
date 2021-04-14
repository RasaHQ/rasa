import dask

from rasa.architecture_prototype import graph
from rasa.architecture_prototype.graph import (
    DomainReader,
    TrainingDataReader,
    StoryToTrainingDataConverter,
    StoryGraphReader,
    MessageToE2EFeatureConverter,
    TrackerGenerator,
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

# We can omit `FallbackClassifier` as this doesn't train
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
        "uses": StoryGraphReader,
        "fn": "read",
        "config": {"project": "examples/moodbot"},
        "needs": {},
    },
    "generate_trackers": {
        "uses": TrackerGenerator,
        "fn": "generate",
        "config": {},
        "needs": {"domain": "load_domain", "story_graph": "load_stories"},
    },
    "convert_stories_for_nlu": {
        "uses": StoryToTrainingDataConverter,
        "fn": "convert",
        "config": {},
        "needs": {"story_graph": "load_stories"},
    },
    "core_tokenize": {
        "uses": WhitespaceTokenizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "convert_stories_for_nlu"},
    },
    "core_train_regex_featurizer": {
        "uses": RegexFeaturizer,
        "fn": "train",
        "config": {
            "component_config": {
                "model_dir": "model",
                "filename": "core_regex_featurizer",
            }
        },
        "needs": {"training_data": "core_tokenize"},
    },
    "core_add_regex_features": {
        "uses": RegexFeaturizer,
        "fn": "process_training_data",
        "config": {},
        "needs": {
            "filename": "core_train_regex_featurizer",
            "training_data": "core_tokenize",
        },
    },
    "core_train_lexical_featurizer": {
        "uses": LexicalSyntacticFeaturizer,
        "fn": "train",
        "config": {
            "component_config": {
                "model_dir": "model",
                "filename": "core_lexical_featurizer",
            }
        },
        "needs": {"training_data": "core_tokenize"},
    },
    "core_add_lexical_features": {
        "uses": LexicalSyntacticFeaturizer,
        "fn": "process_training_data",
        "config": {
            "component_config": {
                "model_dir": "model",
                "filename": "core_lexical_featurizer",
            }
        },
        "needs": {
            "training_data": "core_add_regex_features",
            "filename": "core_train_lexical_featurizer",
        },
    },
    "core_train_count_featurizer1": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {
            "component_config": {
                "model_dir": "model",
                "filename": "core_count_featurizer1",
            }
        },
        "needs": {"training_data": "core_tokenize"},
    },
    "core_add_count_features1": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "eager": False,
        "fn": "process_training_data",
        "config": {
            "component_config": {
                "model_dir": "model",
                "filename": "core_count_featurizer1",
            }
        },
        "needs": {
            "training_data": "core_add_lexical_features",
            "filename": "core_train_count_featurizer1",
        },
    },
    "core_train_count_featurizer2": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {
            "component_config": {
                "model_dir": "model",
                "filename": "core_count_featurizer2",
            }
        },
        "needs": {"training_data": "core_tokenize"},
    },
    "core_add_count_features2": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "eager": False,
        "fn": "process_training_data",
        "config": {
            "component_config": {
                "model_dir": "model",
                "filename": "core_count_featurizer2",
            }
        },
        "needs": {
            "filename": "core_train_count_featurizer2",
            "training_data": "core_add_count_features1",
        },
    },
    "create_e2e_lookup": {
        "uses": MessageToE2EFeatureConverter,
        "fn": "convert",
        "config": {},
        "needs": {"training_data": "core_add_count_features2",},
    },
    "train_memoization_policy": {
        "uses": MemoizationPolicy,
        "fn": "train",
        "config": {},
        "needs": {"training_trackers": "generate_trackers", "domain": "load_domain"},
    },
    "train_rule_policy": {
        "uses": RulePolicy,
        "fn": "train",
        "config": {},
        "needs": {"training_trackers": "generate_trackers", "domain": "load_domain"},
    },
    "train_ted_policy": {
        "uses": TEDPolicy,
        "fn": "train",
        "config": {},
        "needs": {
            "e2e_features": "create_e2e_lookup",
            "training_trackers": "generate_trackers",
            "domain": "load_domain",
        },
    },
    **rasa_nlu_train_graph,
}


def test_visualize_e2e_graph():
    dask_graph = graph.convert_to_dask_graph(full_model_train_graph)

    dask.visualize(dask_graph, filename="e2e_graph.png")


def test_train_full_model():
    core_targets = ["train_memoization_policy", "train_ted_policy", "train_rule_policy"]
    nlu_targets = [
        "train_classifier",
        "train_response_selector",
        "train_synonym_mapper",
    ]
    trained_components = graph.run_as_dask_graph(
        full_model_train_graph, core_targets + nlu_targets
    )

    print(trained_components)
