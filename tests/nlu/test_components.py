import pytest
from rasa.nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor

from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor

from rasa.nlu import components
from rasa.nlu.extractors.duckling_http_extractor import DucklingHTTPExtractor
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import CountVectorsFeaturizer
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.nlu.training_data import TrainingData, Message


@pytest.mark.parametrize(
    "components_list,training_data",
    [
        (
            [DucklingHTTPExtractor()],
            TrainingData(
                [Message("", {"entities": [{"start": 0, "end": 7, "entity": "snake"}]})]
            ),
        )
    ],
    [
        (
            [DucklingHTTPExtractor()],
            TrainingData(
                [Message("", {"entities": [{"start": 0, "end": 7, "entity": "snake"}]})]
            ),
        )
    ],
)
def test_validate_required_components_from_data_missing(
    components_list, training_data
):
    with pytest.warns(UserWarning):
        components.validate_required_components_from_data(
            components_list, training_data
        )
