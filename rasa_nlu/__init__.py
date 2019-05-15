import rasa.nlu
import sys
import warnings

# this makes sure old code can still import from `rasa_nlu`
# although the package has been moved to `rasa.nlu`
sys.modules["rasa_nlu"] = rasa.nlu

warnings.warn(
    "The 'rasa_nlu' package has been renamed. You should change "
    "your imports to use 'rasa.nlu' instead.",
    UserWarning,
)
