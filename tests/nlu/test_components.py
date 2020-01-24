import pytest

from rasa.nlu import components
from rasa.nlu.extractors.duckling_http_extractor import DucklingHTTPExtractor
from rasa.nlu.training_data import TrainingData, Message


def test_validate_required_components_from_data_missing_crf():
    components_list = [DucklingHTTPExtractor()]
    training_data = TrainingData([Message('', {'entities': [{'start': 0, 'end': 7, 'entity': 'apple'}]})])
    with pytest.warns(UserWarning):
        components.validate_required_components_from_data(components_list, training_data)

