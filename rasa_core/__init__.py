import logging

import rasa

from rasa_core.train import train
from rasa_core.test import test
from rasa_core.visualize import visualize

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = rasa.__version__
