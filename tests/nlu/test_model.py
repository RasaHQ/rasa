import pickle

from rasa.nlu.model import UnsupportedModelError


def test_exception_pickling():
    exception = UnsupportedModelError("test run")
    cycled_exception = pickle.loads(pickle.dumps(exception))
    assert exception.message == cycled_exception.message
