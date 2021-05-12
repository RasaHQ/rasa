from rasa.architecture_prototype.graph_components import (
    DomainReader,
    MessageCreator,
    ProjectProvider,
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

train_graph_schema = {
    "get_project": {
        "uses": ProjectProvider,
        "fn": "get",
        "config": {"project": None},
        "needs": {},
        "persistor": False,
    },
    "load_domain": {
        "uses": DomainReader,
        "fn": "read",
        "config": {},
        "needs": {"project": "get_project"},
    },
    "load_stories": {
        "uses": StoryGraphReader,
        "fn": "read",
        "config": {},
        "needs": {"project": "get_project"},
        "persistor": False,
    },
    "generate_trackers": {
        "uses": TrackerGenerator,
        "fn": "generate",
        "config": {},
        "needs": {"domain": "load_domain", "story_graph": "load_stories"},
        "persistor": False,
    },
    "MemoizationPolicy_0": {
        "uses": MemoizationPolicy,
        "fn": "train",
        "config": {},
        "needs": {"training_trackers": "generate_trackers", "domain": "load_domain"},
    },
    "TEDPolicy_1": {
        "uses": TEDPolicy,
        "fn": "train",
        "config": {"max_history": 5, "epochs": 1, "constrain_similarities": True},
        "needs": {
            "training_trackers": "generate_trackers",
            "domain": "load_domain",
            "e2e_features": "create_e2e_lookup",
        },
    },
    "RulePolicy_2": {
        "uses": RulePolicy,
        "fn": "train",
        "config": {},
        "needs": {"training_trackers": "generate_trackers", "domain": "load_domain"},
    },
    "convert_stories_for_nlu": {
        "uses": StoryToTrainingDataConverter,
        "fn": "convert_for_training",
        "config": {},
        "needs": {"story_graph": "load_stories", "domain": "load_domain"},
        "persistor": False,
    },
    "process_core_WhitespaceTokenizer_0": {
        "uses": WhitespaceTokenizer,
        "fn": "process_training_data",
        "config": {},
        "needs": {"training_data": "convert_stories_for_nlu"},
        "persistor": False,
    },
    "train_core_RegexFeaturizer_1": {
        "uses": RegexFeaturizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "process_core_WhitespaceTokenizer_0"},
    },
    "process_core_RegexFeaturizer_1": {
        "uses": RegexFeaturizer,
        "fn": "process_training_data",
        "config": {},
        "needs": {
            "resource_name": "train_core_RegexFeaturizer_1",
            "training_data": "process_core_WhitespaceTokenizer_0",
        },
    },
    "train_core_LexicalSyntacticFeaturizer_2": {
        "uses": LexicalSyntacticFeaturizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "process_core_RegexFeaturizer_1"},
    },
    "process_core_LexicalSyntacticFeaturizer_2": {
        "uses": LexicalSyntacticFeaturizer,
        "fn": "process_training_data",
        "config": {},
        "needs": {
            "resource_name": "train_core_LexicalSyntacticFeaturizer_2",
            "training_data": "process_core_RegexFeaturizer_1",
        },
    },
    "train_core_CountVectorsFeaturizer_3": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "process_core_LexicalSyntacticFeaturizer_2"},
    },
    "process_core_CountVectorsFeaturizer_3": {
        "uses": CountVectorsFeaturizer,
        "fn": "process_training_data",
        "config": {},
        "needs": {
            "resource_name": "train_core_CountVectorsFeaturizer_3",
            "training_data": "process_core_LexicalSyntacticFeaturizer_2",
        },
    },
    "train_core_CountVectorsFeaturizer_4": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {"analyzer": "char_wb", "min_ngram": 1, "max_ngram": 4},
        "needs": {"training_data": "process_core_CountVectorsFeaturizer_3"},
    },
    "process_core_CountVectorsFeaturizer_4": {
        "uses": CountVectorsFeaturizer,
        "fn": "process_training_data",
        "config": {"analyzer": "char_wb", "min_ngram": 1, "max_ngram": 4},
        "needs": {
            "resource_name": "train_core_CountVectorsFeaturizer_4",
            "training_data": "process_core_CountVectorsFeaturizer_3",
        },
    },
    "create_e2e_lookup": {
        "uses": MessageToE2EFeatureConverter,
        "fn": "convert",
        "config": {},
        "needs": {"messages": "process_core_CountVectorsFeaturizer_4"},
        "persistor": False,
    },
    "load_data": {
        "uses": TrainingDataReader,
        "fn": "read",
        "config": {},
        "needs": {"project": "get_project"},
        "persistor": False,
    },
    "process_WhitespaceTokenizer_0": {
        "uses": WhitespaceTokenizer,
        "fn": "process_training_data",
        "config": {},
        "needs": {"training_data": "load_data"},
        "persistor": False,
    },
    "train_RegexFeaturizer_1": {
        "uses": RegexFeaturizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "process_WhitespaceTokenizer_0"},
    },
    "process_RegexFeaturizer_1": {
        "uses": RegexFeaturizer,
        "fn": "process_training_data",
        "config": {},
        "needs": {
            "resource_name": "train_RegexFeaturizer_1",
            "training_data": "process_WhitespaceTokenizer_0",
        },
    },
    "train_LexicalSyntacticFeaturizer_2": {
        "uses": LexicalSyntacticFeaturizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "process_RegexFeaturizer_1"},
    },
    "process_LexicalSyntacticFeaturizer_2": {
        "uses": LexicalSyntacticFeaturizer,
        "fn": "process_training_data",
        "config": {},
        "needs": {
            "resource_name": "train_LexicalSyntacticFeaturizer_2",
            "training_data": "process_RegexFeaturizer_1",
        },
    },
    "train_CountVectorsFeaturizer_3": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "process_LexicalSyntacticFeaturizer_2"},
    },
    "process_CountVectorsFeaturizer_3": {
        "uses": CountVectorsFeaturizer,
        "fn": "process_training_data",
        "config": {},
        "needs": {
            "resource_name": "train_CountVectorsFeaturizer_3",
            "training_data": "process_LexicalSyntacticFeaturizer_2",
        },
    },
    "train_CountVectorsFeaturizer_4": {
        "uses": CountVectorsFeaturizer,
        "fn": "train",
        "config": {"analyzer": "char_wb", "min_ngram": 1, "max_ngram": 4},
        "needs": {"training_data": "process_CountVectorsFeaturizer_3"},
    },
    "process_CountVectorsFeaturizer_4": {
        "uses": CountVectorsFeaturizer,
        "fn": "process_training_data",
        "config": {"analyzer": "char_wb", "min_ngram": 1, "max_ngram": 4},
        "needs": {
            "resource_name": "train_CountVectorsFeaturizer_4",
            "training_data": "process_CountVectorsFeaturizer_3",
        },
    },
    "train_DIETClassifier_5": {
        "uses": DIETClassifier,
        "fn": "train",
        "config": {"epochs": 1, "constrain_similarities": True},
        "needs": {"training_data": "process_CountVectorsFeaturizer_4"},
    },
    "train_EntitySynonymMapper_6": {
        "uses": EntitySynonymMapper,
        "fn": "train",
        "config": {},
        "needs": {"training_data": "process_CountVectorsFeaturizer_4"},
    },
    "train_ResponseSelector_7": {
        "uses": ResponseSelector,
        "fn": "train",
        "config": {"epochs": 1, "constrain_similarities": True},
        "needs": {"training_data": "process_CountVectorsFeaturizer_4"},
    },
    "targets": [
        "MemoizationPolicy_0",
        "TEDPolicy_1",
        "RulePolicy_2",
        "train_DIETClassifier_5",
        "train_EntitySynonymMapper_6",
        "train_ResponseSelector_7",
    ],
}


predict_graph_schema = {
    "load_user_message": {
        "uses": MessageCreator,
        "fn": "create",
        "config": {"message": None},
        "needs": {},
        "persistor": False,
    },
    "convert_message_to_nlu": {
        "uses": NLUMessageConverter,
        "fn": "convert",
        "config": {},
        "needs": {"message": "load_user_message"},
        "persistor": False,
    },
    "load_history": {
        "uses": TrackerLoader,
        "fn": "load",
        "needs": {},
        "config": {"tracker": None},
        "persistor": False,
    },
    "add_parsed_nlu_message": {
        "uses": NLUPredictionToHistoryAdder,
        "fn": "merge",
        "needs": {
            "tracker": "load_history",
            "initial_user_message": "load_user_message",
            "parsed_messages": "FallbackClassifier_8",
            "domain": "load_domain",
        },
        "config": {},
        "persistor": False,
    },
    "load_domain": {
        "uses": DomainReader,
        "constructor_name": "load",
        "fn": "provide",
        "config": {"resource_name": "load_domain"},
        "needs": {},
    },
    "select_prediction": {
        "uses": SimplePolicyEnsemble,
        "fn": "probabilities_using_best_policy",
        "config": {},
        "persistor": False,
        "needs": {
            "tracker": "add_parsed_nlu_message",
            "domain": "load_domain",
            "MemoizationPolicy_0_prediction": "MemoizationPolicy_0",
            "TEDPolicy_1_prediction": "TEDPolicy_1",
            "RulePolicy_2_prediction": "RulePolicy_2",
        },
    },
    "WhitespaceTokenizer_0": {
        "uses": WhitespaceTokenizer,
        "constructor_name": "load",
        "fn": "process",
        "config": {"resource_name": "train_WhitespaceTokenizer_0"},
        "needs": {"messages": "convert_message_to_nlu"},
    },
    "RegexFeaturizer_1": {
        "uses": RegexFeaturizer,
        "constructor_name": "load",
        "fn": "process",
        "config": {"resource_name": "train_RegexFeaturizer_1"},
        "needs": {"messages": "WhitespaceTokenizer_0"},
    },
    "LexicalSyntacticFeaturizer_2": {
        "uses": LexicalSyntacticFeaturizer,
        "constructor_name": "load",
        "fn": "process",
        "config": {"resource_name": "train_LexicalSyntacticFeaturizer_2"},
        "needs": {"messages": "RegexFeaturizer_1"},
    },
    "CountVectorsFeaturizer_3": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "fn": "process",
        "config": {"resource_name": "train_CountVectorsFeaturizer_3"},
        "needs": {"messages": "LexicalSyntacticFeaturizer_2"},
    },
    "CountVectorsFeaturizer_4": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "fn": "process",
        "config": {
            "resource_name": "train_CountVectorsFeaturizer_4",
            "analyzer": "char_wb",
            "min_ngram": 1,
            "max_ngram": 4,
        },
        "needs": {"messages": "CountVectorsFeaturizer_3"},
    },
    "DIETClassifier_5": {
        "uses": DIETClassifier,
        "constructor_name": "load",
        "fn": "process",
        "config": {
            "resource_name": "train_DIETClassifier_5",
            "epochs": 1,
            "constrain_similarities": True,
        },
        "needs": {"messages": "CountVectorsFeaturizer_4"},
    },
    "EntitySynonymMapper_6": {
        "uses": EntitySynonymMapper,
        "constructor_name": "load",
        "fn": "process",
        "config": {"resource_name": "train_EntitySynonymMapper_6"},
        "needs": {"messages": "DIETClassifier_5"},
    },
    "ResponseSelector_7": {
        "uses": ResponseSelector,
        "constructor_name": "load",
        "fn": "process",
        "config": {
            "resource_name": "train_ResponseSelector_7",
            "epochs": 1,
            "constrain_similarities": True,
        },
        "needs": {"messages": "EntitySynonymMapper_6"},
    },
    "FallbackClassifier_8": {
        "config": {
            "ambiguity_threshold": 0.1,
            "resource_name": "train_FallbackClassifier_8",
            "threshold": 0.3,
        },
        "constructor_name": "load",
        "fn": "process",
        "needs": {"messages": "ResponseSelector_7"},
        "uses": FallbackClassifier,
    },
    "MemoizationPolicy_0": {
        "uses": MemoizationPolicy,
        "constructor_name": "load",
        "fn": "predict_action_probabilities",
        "config": {"resource_name": "MemoizationPolicy_0"},
        "needs": {"tracker": "add_parsed_nlu_message", "domain": "load_domain"},
    },
    "TEDPolicy_1": {
        "uses": TEDPolicy,
        "constructor_name": "load",
        "fn": "predict_action_probabilities",
        "config": {
            "resource_name": "TEDPolicy_1",
            "max_history": 5,
            "epochs": 1,
            "constrain_similarities": True,
        },
        "needs": {
            "tracker": "add_parsed_nlu_message",
            "domain": "load_domain",
            "e2e_features": "create_e2e_lookup",
        },
    },
    "RulePolicy_2": {
        "uses": RulePolicy,
        "constructor_name": "load",
        "fn": "predict_action_probabilities",
        "config": {"resource_name": "RulePolicy_2"},
        "needs": {"tracker": "add_parsed_nlu_message", "domain": "load_domain"},
    },
    "convert_tracker_for_e2e": {
        "uses": StoryToTrainingDataConverter,
        "fn": "convert_for_inference",
        "config": {},
        "needs": {"tracker": "add_parsed_nlu_message"},
        "persistor": False,
    },
    "create_e2e_lookup": {
        "uses": MessageToE2EFeatureConverter,
        "fn": "convert",
        "config": {},
        "needs": {"messages": "core_CountVectorsFeaturizer_4"},
        "persistor": False,
    },
    "core_WhitespaceTokenizer_0": {
        "uses": WhitespaceTokenizer,
        "constructor_name": "load",
        "fn": "process",
        "config": {"resource_name": "train_core_WhitespaceTokenizer_0"},
        "needs": {"messages": "convert_tracker_for_e2e"},
    },
    "core_RegexFeaturizer_1": {
        "uses": RegexFeaturizer,
        "constructor_name": "load",
        "fn": "process",
        "config": {"resource_name": "train_core_RegexFeaturizer_1"},
        "needs": {"messages": "core_WhitespaceTokenizer_0"},
    },
    "core_LexicalSyntacticFeaturizer_2": {
        "uses": LexicalSyntacticFeaturizer,
        "constructor_name": "load",
        "fn": "process",
        "config": {"resource_name": "train_core_LexicalSyntacticFeaturizer_2"},
        "needs": {"messages": "core_RegexFeaturizer_1"},
    },
    "core_CountVectorsFeaturizer_3": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "fn": "process",
        "config": {"resource_name": "train_core_CountVectorsFeaturizer_3"},
        "needs": {"messages": "core_LexicalSyntacticFeaturizer_2"},
    },
    "core_CountVectorsFeaturizer_4": {
        "uses": CountVectorsFeaturizer,
        "constructor_name": "load",
        "fn": "process",
        "config": {
            "resource_name": "train_core_CountVectorsFeaturizer_4",
            "analyzer": "char_wb",
            "min_ngram": 1,
            "max_ngram": 4,
        },
        "needs": {"messages": "core_CountVectorsFeaturizer_3"},
    },
    "targets": ["select_prediction"],
}
