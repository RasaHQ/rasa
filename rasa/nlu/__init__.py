import logging

import rasa
from rasa.nlu.train import train
from rasa.nlu.test import run_evaluation as test
from rasa.nlu.test import cross_validate

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = rasa.__version__
