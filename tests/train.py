from rasa.utils.tensorflow.constants import EPOCHS
from typing import Any, Dict, List, Tuple, Text, Union


COMPONENTS_TEST_PARAMS = {
    "DIETClassifier": {EPOCHS: 1},
    "ResponseSelector": {EPOCHS: 1},
    "LanguageModelFeaturizer": {
        "model_name": "bert",
        "model_weights": "bert-base-uncased",
    },
}


def get_test_params_for_component(component: Text) -> Dict[Text, Union[Text, int]]:
    return (
        COMPONENTS_TEST_PARAMS[component] if component in COMPONENTS_TEST_PARAMS else {}
    )


def as_pipeline(*components):
    return [{**{"name": c}, **get_test_params_for_component(c)} for c in components]


def pipelines_for_tests() -> List[Tuple[Text, List[Dict[Text, Any]]]]:
    # these templates really are just for testing
    # every component should be in here so train-persist-load-use cycle can be
    # tested they still need to be in a useful order - hence we can not simply
    # generate this automatically.

    # Create separate test pipelines for dense featurizers
    # because they can't co-exist in the same pipeline together,
    # as their tokenizers break the incoming message into different number of tokens.

    # first is language followed by list of components
    return [
        ("en", as_pipeline("KeywordIntentClassifier")),
        (
            "en",
            as_pipeline(
                "WhitespaceTokenizer",
                "RegexFeaturizer",
                "LexicalSyntacticFeaturizer",
                "CountVectorsFeaturizer",
                "CRFEntityExtractor",
                "DucklingEntityExtractor",
                "DIETClassifier",
                "ResponseSelector",
                "EntitySynonymMapper",
            ),
        ),
        (
            "en",
            as_pipeline(
                "SpacyNLP",
                "SpacyTokenizer",
                "SpacyFeaturizer",
                "SpacyEntityExtractor",
                "SklearnIntentClassifier",
            ),
        ),
        (
            "en",
            as_pipeline(
                "WhitespaceTokenizer", "LanguageModelFeaturizer", "DIETClassifier"
            ),
        ),
        ("fallback", as_pipeline("KeywordIntentClassifier", "FallbackClassifier")),
    ]


def pipelines_for_non_windows_tests() -> List[Tuple[Text, List[Dict[Text, Any]]]]:
    # these templates really are just for testing

    # because some of the components are not available on Windows, we specify pipelines
    # containing them separately

    # first is language followed by list of components
    return [
        (
            "en",
            as_pipeline(
                "SpacyNLP", "SpacyTokenizer", "SpacyFeaturizer", "DIETClassifier"
            ),
        ),
        (
            "en",
            as_pipeline(
                "MitieNLP",
                "MitieTokenizer",
                "MitieFeaturizer",
                "MitieIntentClassifier",
                "RegexEntityExtractor",
            ),
        ),
        (
            "zh",
            as_pipeline(
                "MitieNLP", "JiebaTokenizer", "MitieFeaturizer", "MitieEntityExtractor"
            ),
        ),
    ]
