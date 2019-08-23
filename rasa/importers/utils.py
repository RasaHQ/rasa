from typing import Iterable, Text

from rasa.nlu.training_data.training_data import TrainingData


def training_data_from_paths(paths: Iterable[Text], language: Text) -> TrainingData:
    from rasa.nlu.training_data.data_manager import DataManager

    training_datas = [DataManager.load_data(nlu_file, language) for nlu_file in paths]
    return TrainingData().merge(*training_datas)
