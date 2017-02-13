import pytest
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.trainers.mitie_trainer import MITIETrainer


class TestMitieTrainer:
    def test_failure_on_invalid_lang(self):
        config = RasaNLUConfig("config_mitie.json")
        with pytest.raises(NotImplementedError):
            MITIETrainer(config['mitie_file'], 'umpalumpa')
