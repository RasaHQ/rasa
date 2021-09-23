import logging

import rasa

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = rasa.__version__
