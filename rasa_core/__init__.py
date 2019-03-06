import logging

import rasa_core.version

from rasa_core.train import train
from rasa_core.test import test
from rasa_core.visualize import visualize

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = rasa_core.version.__version__
