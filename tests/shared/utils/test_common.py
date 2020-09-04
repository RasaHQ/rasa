import random

import rasa.shared.utils.common


def test_all_subclasses():
    class TestClass:
        pass

    classes = [type(f"TestClass{i}", (TestClass,), {}) for i in range(10)]

    assert rasa.shared.utils.common.all_subclasses(TestClass) == classes
