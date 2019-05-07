import logging

import rasa.version

from rasa.run import run
from rasa.train import train
from rasa.test import test

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = rasa.version.__version__
