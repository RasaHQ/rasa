import glob
from typing import Dict, Text, Optional, Any

import pytest

import rasa.core.config
from rasa.core.constants import (
    DEFAULT_NLU_FALLBACK_THRESHOLD,
    DEFAULT_CORE_FALLBACK_THRESHOLD,
    DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
    DEFAULT_MAX_HISTORY,
)
from rasa.core.policies.ensemble import PolicyEnsemble
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.shared.core.constants import (
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_TWO_STAGE_FALLBACK_NAME,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
    YAMLStoryWriter,
)
import rasa.shared.utils.io
from tests.core.conftest import ExamplePolicy


@pytest.mark.parametrize("filename", glob.glob("data/test_config/example_config.yaml"))
def test_load_config(filename):
    loaded = rasa.core.config.load(filename)
    assert len(loaded) == 2
    assert isinstance(loaded[0], MemoizationPolicy)
    assert isinstance(loaded[1], ExamplePolicy)
