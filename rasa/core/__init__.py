import logging

import rasa

from rasa.core.train import train
from rasa.core.visualize import visualize

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = rasa.__version__
