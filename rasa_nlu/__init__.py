import logging

import rasa_nlu.version

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = rasa_nlu.version.__version__
