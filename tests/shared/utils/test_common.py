import rasa.shared.utils.common


def test_all_subclasses():
    class TestClass:
        pass

    subclasses = [type(f"Subclass{i}", (TestClass,), {}) for i in range(10)]
    sub_subclasses = [
        type(f"Sub-subclass_{subclass.__name__}", (subclass,), {})
        for subclass in subclasses
    ]

    expected = subclasses + sub_subclasses
    assert rasa.shared.utils.common.all_subclasses(TestClass) == expected
