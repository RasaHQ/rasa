from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import pytest

from tests.conftest import ExamplePolicy
from rasa_core.config import load
from rasa_core.policies.memoization import MemoizationPolicy


@pytest.mark.parametrize("filename", glob.glob(
            "data/test_config/example_config.yaml"))
def test_load_config(filename):
    loaded = load(filename)
    assert len(loaded) == 2
    assert isinstance(loaded[0], MemoizationPolicy)
    assert isinstance(loaded[1], ExamplePolicy)
