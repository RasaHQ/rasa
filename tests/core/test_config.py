import glob
import pytest

import rasa.core.config
from rasa.core.policies.memoization import MemoizationPolicy
import rasa.shared.utils.io
from tests.core.conftest import ExamplePolicy


@pytest.mark.parametrize("filename", glob.glob("data/test_config/example_config.yaml"))
def test_load_config(filename):
    loaded = rasa.core.config.load(filename)
    assert len(loaded) == 2
    assert isinstance(loaded[0], MemoizationPolicy)
    assert isinstance(loaded[1], ExamplePolicy)
