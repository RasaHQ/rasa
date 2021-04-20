from rasa.architecture_prototype.graph_components import (
    DomainReader,
    MessageCreator,
    TrainingDataReader,
    StoryToTrainingDataConverter,
    StoryGraphReader,
    MessageToE2EFeatureConverter,
    TrackerGenerator,
    NLUPredictionToHistoryAdder,
    NLUMessageConverter,
    TrackerLoader,
)
from rasa.core.policies import SimplePolicyEnsemble
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.classifiers.fallback_classifier import FallbackClassifier
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
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.trackers import DialogueStateTracker


project = "examples/moodbot"

# We can omit `FallbackClassifier` as this doesn't train
rasa_nlu_train_graph = {
    "load_data": {
        "uses": TrainingDataReader,
        "fn": "read",
        "config": {"project": project},
        "needs": {},
        "persist": False,
    },
    "tokenize": {
        "uses": WhitespaceTokenizer,
        "fn": "process_training_data",
        "config": {},
        "needs": {"training_data": "load_data"},
        "persist": False,
    },
    "train_regex_featurizer": {
        "uses": RegexFeaturizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "tokenize"},
    },
    "add_regex_features": {
        "uses": RegexFeaturizer,
        "fn": "process_training_data",
        "config": {},
        "needs": {
            "resource_name": "train_regex_featurizer",
            "training_data": "tokenize",
        },
    },
    "train_lexical_featurizer": {
        "uses": LexicalSyntacticFeaturizer,
        "fn": "train",
        "config": {"component_config": {}},
        "needs": {"training_data": "tokenize"},
    },
    "add_lexical_features": {
        "uses": LexicalSyntacticFeaturizer,
        "fn": "process_training_data",
        "config": {"component_config": {}},
        "needs": {
            "training_data": "add_regex_features",
            "resource_name": "train_lexical_featurizer",
        },
    },
    "train_count_featurizer1": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "tokenize"},
    },
    "add_count_features1": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "eager": False,
        "fn": "process_training_data",
        "config": {},
        "needs": {
            "training_data": "add_lexical_features",
            "resource_name": "train_count_featurizer1",
        },
    },
    "train_count_featurizer2": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "tokenize"},
    },
    "add_count_features2": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "eager": False,
        "fn": "process_training_data",
        "config": {},
        "needs": {
            "resource_name": "train_count_featurizer2",
            "training_data": "add_count_features1",
        },
    },
    "train_classifier": {
        "uses": DIETClassifier,
        "fn": "train",
        "config": {"component_config": {"epochs": 100,},},
        "needs": {"training_data": "add_count_features2"},
    },
    "train_response_selector": {
        "uses": ResponseSelector,
        "fn": "train",
        "config": {"component_config": {"epochs": 1,},},
        "needs": {"training_data": "add_count_features2"},
    },
    "train_synonym_mapper": {
        "uses": EntitySynonymMapper,
        "config": {},
        "fn": "train",
        "needs": {"training_data": "add_count_features2",},
    },
}

full_model_train_graph = {
    "load_domain": {
        "uses": DomainReader,
        "fn": "read",
        "config": {"project": project},
        "needs": {},
    },
    "load_stories": {
        "uses": StoryGraphReader,
        "fn": "read",
        "config": {"project": project},
        "needs": {},
        "persist": False,
    },
    "generate_trackers": {
        "uses": TrackerGenerator,
        "fn": "generate",
        "config": {},
        "needs": {"domain": "load_domain", "story_graph": "load_stories"},
        "persist": False,
    },
    "convert_stories_for_nlu": {
        "uses": StoryToTrainingDataConverter,
        "fn": "convert",
        "config": {},
        "needs": {"story_graph": "load_stories"},
        "persist": False,
    },
    "core_tokenize": {
        "uses": WhitespaceTokenizer,
        "fn": "process_training_data",
        "config": {},
        "needs": {"training_data": "convert_stories_for_nlu"},
        "persist": False,
    },
    "core_train_regex_featurizer": {
        "uses": RegexFeaturizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "core_tokenize"},
    },
    "core_add_regex_features": {
        "uses": RegexFeaturizer,
        "fn": "process_training_data",
        "config": {},
        "needs": {
            "resource_name": "core_train_regex_featurizer",
            "training_data": "core_tokenize",
        },
    },
    "core_train_lexical_featurizer": {
        "uses": LexicalSyntacticFeaturizer,
        "fn": "train",
        "config": {"component_config": {}},
        "needs": {"training_data": "core_tokenize"},
    },
    "core_add_lexical_features": {
        "uses": LexicalSyntacticFeaturizer,
        "fn": "process_training_data",
        "config": {"component_config": {}},
        "needs": {
            "training_data": "core_add_regex_features",
            "resource_name": "core_train_lexical_featurizer",
        },
    },
    "core_train_count_featurizer1": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "core_tokenize"},
    },
    "core_add_count_features1": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "eager": False,
        "fn": "process_training_data",
        "config": {},
        "needs": {
            "training_data": "core_add_lexical_features",
            "resource_name": "core_train_count_featurizer1",
        },
    },
    "core_train_count_featurizer2": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "core_tokenize"},
    },
    "core_add_count_features2": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "eager": False,
        "fn": "process_training_data",
        "config": {},
        "needs": {
            "resource_name": "core_train_count_featurizer2",
            "training_data": "core_add_count_features1",
        },
    },
    "create_e2e_lookup": {
        "uses": MessageToE2EFeatureConverter,
        "fn": "convert",
        "config": {},
        "needs": {"training_data": "core_add_count_features2",},
        "persist": False,
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
        "config": {"max_history": 5, "checkpoint_model": True},
        "needs": {
            "e2e_features": "create_e2e_lookup",
            "training_trackers": "generate_trackers",
            "domain": "load_domain",
        },
    },
    **rasa_nlu_train_graph,
}

predict_graph_schema = {
    "load_user_message": {
        "uses": MessageCreator,
        "fn": "create",
        "config": {"text": "Hello"},
        "needs": {},
        "persist": False,
    },
    "convert_message_to_nlu": {
        "uses": NLUMessageConverter,
        "fn": "convert",
        "config": {},
        "needs": {"message": "load_user_message"},
        "persist": False,
    },
    "tokenize": {
        "uses": WhitespaceTokenizer,
        "fn": "process",
        "config": {},
        "needs": {"message": "convert_message_to_nlu"},
        "persist": False,
    },
    "add_regex_features": {
        "uses": RegexFeaturizer,
        "constructor_name": "load",
        "fn": "process",
        "config": {"resource_name": "train_regex_featurizer"},
        "needs": {"message": "tokenize",},
    },
    "add_lexical_features": {
        "uses": LexicalSyntacticFeaturizer,
        "constructor_name": "load",
        "fn": "process",
        "config": {"resource_name": "train_lexical_featurizer", "component_config": {}},
        "needs": {"message": "add_regex_features",},
    },
    "add_count_features1": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "fn": "process",
        "config": {"resource_name": "train_count_featurizer1",},
        "needs": {"message": "add_lexical_features",},
    },
    "add_count_features2": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "fn": "process",
        "config": {"resource_name": "train_count_featurizer2",},
        "needs": {"message": "add_count_features1",},
    },
    "classify": {
        "uses": DIETClassifier,
        "fn": "process",
        "constructor_name": "load",
        "config": {
            "component_config": {"epochs": 1,},
            "resource_name": "train_classifier",
        },
        "needs": {"message": "add_count_features2"},
    },
    "synonym_mapper": {
        "uses": EntitySynonymMapper,
        "constructor_name": "load",
        "fn": "process",
        "config": {"resource_name": "train_synonym_mapper"},
        "needs": {"message": "classify",},
    },
    "response_selector": {
        "uses": ResponseSelector,
        "constructor_name": "load",
        "fn": "process",
        "config": {
            "resource_name": "train_response_selector",
            "component_config": {"epochs": 1,},
        },
        "needs": {"message": "synonym_mapper"},
    },
    "fallback_classifier": {
        "uses": FallbackClassifier,
        "fn": "process",
        "config": {},
        "needs": {"message": "response_selector"},
    },
    "load_history": {
        "uses": TrackerLoader,
        "fn": "load",
        "needs": {},
        "config": {
            "tracker": DialogueStateTracker.from_events(
                "some_sender", [ActionExecuted(action_name=ACTION_LISTEN_NAME)]
            )
        },
        "persist": False,
    },
    "load_domain": {
        "uses": DomainReader,
        "fn": "provide",
        "config": {"resource_name": "load_domain"},
        "needs": {},
    },
    "add_parsed_nlu_message": {
        "uses": NLUPredictionToHistoryAdder,
        "fn": "merge",
        "needs": {
            "tracker": "load_history",
            "initial_user_message": "load_user_message",
            "parsed_message": "fallback_classifier",
            "domain": "load_domain",
        },
        "config": {},
        "persist": False,
    },
    "predict_memoization_policy": {
        "uses": MemoizationPolicy,
        "constructor_name": "load",
        "fn": "predict_action_probabilities",
        "config": {"resource_name": "train_memoization_policy"},
        "needs": {"tracker": "add_parsed_nlu_message", "domain": "load_domain"},
    },
    "predict_rule_policy": {
        "uses": RulePolicy,
        "constructor_name": "load",
        "fn": "predict_action_probabilities",
        "config": {"resource_name": "train_rule_policy"},
        "needs": {"tracker": "add_parsed_nlu_message", "domain": "load_domain"},
    },
    "predict_ted_policy": {
        "uses": TEDPolicy,
        "constructor_name": "load",
        "fn": "predict_action_probabilities",
        "config": {
            "max_history": 5,
            "checkpoint_model": True,
            "resource_name": "train_ted_policy",
        },
        "needs": {"tracker": "add_parsed_nlu_message", "domain": "load_domain"},
    },
    "select_prediction": {
        "uses": SimplePolicyEnsemble,
        "fn": "probabilities_using_best_policy",
        "config": {},
        "persist": False,
        "needs": {
            "tracker": "add_parsed_nlu_message",
            "domain": "load_domain",
            "rule_prediction": "predict_rule_policy",
            "memo_prediction": "predict_memoization_policy",
            "ted_prediction": "predict_ted_policy",
        },
    },
}
