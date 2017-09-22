from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import six

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.converters import load_data
from rasa_nlu.model import Trainer

if six.PY2:
    model_name = 'current_py2'
else:
    model_name = 'current_py3'


def train_babi_nlu():
    training_data = load_data('examples/babi/data/franken_data.json')
    trainer = Trainer(RasaNLUConfig("examples/babi/data/config_nlu.json"))
    trainer.train(training_data)
    model_directory = trainer.persist('examples/babi/models/nlu/',
                                      model_name=model_name)
    return model_directory


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    train_babi_nlu()
