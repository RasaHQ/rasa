from typing import Iterable, Text

from rasa.nlu.training_data import TrainingData


def training_data_from_paths(paths: Iterable[Text], language: Text) -> TrainingData:
    from rasa.nlu.training_data import loading

    training_datas = [loading.load_data(nlu_file, language) for nlu_file in paths]
    merged_training_data = TrainingData().merge(*training_datas)
    merged_training_data.fill_response_phrases()
    return merged_training_data
