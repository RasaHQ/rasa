import logging

import rasa_nlu.version

from rasa_nlu.train import train
from rasa_nlu.test import run_evaluation as test
from rasa_nlu.test import cross_validate
from rasa_nlu.training_data import load_data

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = rasa_nlu.version.__version__
