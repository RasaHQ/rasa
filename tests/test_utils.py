from rasa_core.utils import is_int


def test_is_int():
    assert is_int(1)
    assert is_int(1.0)
    assert not is_int(None)
    assert not is_int(1.2)
    assert not is_int("test")
