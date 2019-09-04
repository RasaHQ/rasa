import logging

from rasa import version

# define the version before the other imports since these need it
__version__ = version.__version__

from rasa.run import run
from rasa.train import train
from rasa.test import test

logging.getLogger(__name__).addHandler(logging.NullHandler())

# remove handler for abseil,
# a fix for tf 1.14 double logging suggested in
# https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
# from
# https://github.com/dhalperi/pybatfish/blob/f8ddd3938148f9a5d9c14c371a099802c564fac3/pybatfish/client/capirca.py#L33-L50
try:
    import absl.logging

    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass
