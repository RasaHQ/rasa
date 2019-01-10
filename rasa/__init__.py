import logging

import rasa.version

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = rasa.version.__version__
