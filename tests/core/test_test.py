from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

import rasa.core.test
import rasa.nlu.model
import rasa.shared.nlu.interpreter
import rasa.core.interpreter
from _pytest.capture import CaptureFixture
from rasa.core.agent import Agent
from tests.conftest import AsyncMock


async def test_testing_warns_if_action_unknown(
    capsys: CaptureFixture,
    e2e_bot_agent: Agent,
    e2e_bot_test_stories_with_unknown_bot_utterances: Path,
):
    await rasa.core.test.test(
        e2e_bot_test_stories_with_unknown_bot_utterances, e2e_bot_agent
    )
    output = capsys.readouterr().out
    assert "Test story" in output
    assert "contains the bot utterance" in output
    assert "which is not part of the training data / domain" in output


async def test_testing_valid_with_non_e2e_core_model(core_agent: Agent, monkeypatch: MonkeyPatch):
    monkeypatch.setattr(rasa.nlu.model.Interpreter, "featurize_message", None)
    result = await rasa.core.test.test(
        "data/test_yaml_stories/test_stories_entity_annotations.yml", core_agent
    )
    assert "report" in result.keys()
