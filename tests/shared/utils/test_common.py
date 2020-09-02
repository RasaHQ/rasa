import random

import rasa.shared.utils.common


def test_all_subclasses():
    num = random.randint(1, 10)

    class TestClass:
        pass

    classes = [type(f"TestClass{i}", (TestClass,), {}) for i in range(num)]

    assert rasa.shared.utils.common.all_subclasses(TestClass) == classes
