import logging

from rasa import version, plugin  # noqa: F401
from rasa.api import run, train, test  # noqa: F401

# define the version before the other imports since these need it
__version__ = version.__version__


logging.getLogger(__name__).addHandler(logging.NullHandler())
