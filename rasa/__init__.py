import logging
import os

import rasa.version
from rasa.constants import DEFAULT_LOG_LEVEL

from rasa.run import run
from rasa.train import train
from rasa.test import test

log_level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.getLogger(__name__).setLevel(log_level)

__version__ = rasa.version.__version__
