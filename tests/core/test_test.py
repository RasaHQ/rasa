from pathlib import Path
from typing import Text

import pytest

import rasa.core.test
from _pytest.capture import CaptureFixture
from rasa.core.agent import Agent


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


async def test_testing_does_not_warn_if_intent_in_domain(
    default_agent: Agent, stories_path: Text,
):
    with pytest.warns(UserWarning) as record:
        await rasa.core.test.test(Path(stories_path), default_agent)

    assert not any("Found intent" in r.message.args[0] for r in record)
    assert all(
        "in stories which is not part of the domain" not in r.message.args[0]
        for r in record
    )


async def test_testing_valid_with_non_e2e_core_model(core_agent: Agent):
    result = await rasa.core.test.test(
        "data/test_yaml_stories/test_stories_entity_annotations.yml", core_agent
    )
    assert "report" in result.keys()


async def test_action_unlikely_intent_1(
    tmpdir: Path, intent_ted_policy_moodbot_agent: Agent
):
    result = await rasa.core.test.test(
        "data/test_yaml_stories/test_action_unlikely_intent_1.yml",
        intent_ted_policy_moodbot_agent,
        out_directory=str(tmpdir),
    )
    assert "report" in result.keys()
    assert result["report"]["conversation_accuracy"]["correct"] == 1
    assert result["report"]["conversation_accuracy"]["with_warnings"] == 1


async def test_action_unlikely_intent_2(
    tmpdir: Path, intent_ted_policy_moodbot_agent: Agent
):
    result = await rasa.core.test.test(
        "data/test_yaml_stories/test_action_unlikely_intent_2.yml",
        intent_ted_policy_moodbot_agent,
        out_directory=str(tmpdir),
    )
    assert "report" in result.keys()
    assert result["report"]["conversation_accuracy"]["correct"] == 1
    assert result["report"]["conversation_accuracy"]["with_warnings"] == 1


async def test_action_unlikely_intent_complete(
    tmpdir: Path, intent_ted_policy_moodbot_agent: Agent
):
    result = await rasa.core.test.test(
        "data/test_yaml_stories/test_action_unlikely_intent_complete.yml",
        intent_ted_policy_moodbot_agent,
        out_directory=str(tmpdir),
    )
    assert "report" in result.keys()
    assert result["report"]["conversation_accuracy"]["correct"] == 4
    assert result["report"]["conversation_accuracy"]["with_warnings"] == 2
    assert result["report"]["conversation_accuracy"]["total"] == 4


async def test_action_unlikely_intent_wrong_story(
    tmpdir: Path, intent_ted_policy_moodbot_agent: Agent
):
    result = await rasa.core.test.test(
        "data/test_yaml_stories/test_action_unlikely_intent_wrong.yml",
        intent_ted_policy_moodbot_agent,
        out_directory=str(tmpdir),
    )
    assert "report" in result.keys()
    assert result["report"]["conversation_accuracy"]["correct"] == 0
    assert result["report"]["conversation_accuracy"]["with_warnings"] == 0
    assert result["report"]["conversation_accuracy"]["total"] == 1
