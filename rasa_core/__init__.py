import logging

import rasa_core.version

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = rasa_core.version.__version__
