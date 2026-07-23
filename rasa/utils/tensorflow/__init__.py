import importlib.util

# Whether TensorFlow is installed in the current environment. TensorFlow (and the
# rest of the ML/NLU stack) is an optional dependency: it is only installed via the
# ``nlu``/``full`` extras and is not available on Python 3.12+. Components that rely
# on TensorFlow are imported lazily so that a rule-based install (no extras) can run
# on interpreters where TensorFlow has no wheels (e.g. Python 3.13). Use this flag to
# guard any code path that would otherwise import TensorFlow at module load time.
TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
