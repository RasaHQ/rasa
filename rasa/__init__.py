import logging

from rasa import version
from rasa.api import run, train, test

# define the version before the other imports since these need it
__version__ = version.__version__


logging.getLogger(__name__).addHandler(logging.NullHandler())
