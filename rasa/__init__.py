import logging

import rasa.version

# define the version before the other imports since these need it
__version__ = rasa.version.__version__

from rasa.run import run
from rasa.train import train
from rasa.test import test

logging.getLogger(__name__).addHandler(logging.NullHandler())
