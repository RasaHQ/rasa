import rasa.core
import sys
import warnings

# this makes sure old code can still import from `rasa_core`
# although the package has been moved to `rasa.core`
sys.modules["rasa_core"] = rasa.core

warnings.warn(
    "The 'rasa_core' package hase been renamed. You should change "
    "your imports to use 'rasa.core' instead.",
    UserWarning,
)
