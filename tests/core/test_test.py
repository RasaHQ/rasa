from pathlib import Path
from typing import Text

import pytest

import rasa.core.test
from rasa.core.policies.ensemble import SimplePolicyEnsemble
from _pytest.monkeypatch import MonkeyPatch
from _pytest.capture import CaptureFixture
from rasa.core.agent import Agent


def _probabilities_with_action_unlikely_intent_for(intent_name: Text):
    from rasa.core.policies.policy import PolicyPrediction
    from rasa.shared.core.events import UserUttered

    _original = SimplePolicyEnsemble.probabilities_using_best_policy

    def probabilities_using_best_policy(
        self, tracker, domain, interpreter, **kwargs,
    ) -> PolicyPrediction:
        latest_event = tracker.events[-1]
        if (
            isinstance(latest_event, UserUttered)
            and latest_event.parse_data["intent"]["name"] == intent_name
        ):
            # Here we return `action_unlikely_intent` if the name of the latest intent
            # matches `intent_name`.
            # We need to do it because every time the tests are run,
            # training will result in different model weights which will
            # result in different predictions of `action_unlikely_intent`.
            # Because we're not testing `IntentTEDPolicy` here we simply trigger it
            # predicting `action_unlikely_intent` in a specified moment
            # to make the tests deterministic.
            return PolicyPrediction.for_action_name(domain, "action_unlikely_intent")
        return _original(self, tracker, domain, interpreter, **kwargs)

    return probabilities_using_best_policy


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
    monkeypatch: MonkeyPatch, tmp_path: Path, intent_ted_policy_moodbot_agent: Agent
):
    monkeypatch.setattr(
        SimplePolicyEnsemble,
        "probabilities_using_best_policy",
        _probabilities_with_action_unlikely_intent_for("mood_unhappy"),
    )

    file_name = tmp_path / "test_action_unlikely_intent_1.yml"
    file_name.write_text(
        """
        version: "2.0"
        stories:
          - story: unlikely path
            steps:
              - user: |
                  very terrible
                intent: mood_unhappy
              - action: utter_cheer_up
              - action: utter_did_that_help
              - intent: affirm
              - action: utter_happy
        """
    )

    result = await rasa.core.test.test(
        str(file_name), intent_ted_policy_moodbot_agent, out_directory=str(tmp_path),
    )
    assert "report" in result.keys()
    assert result["report"]["conversation_accuracy"]["correct"] == 1
    assert result["report"]["conversation_accuracy"]["with_warnings"] == 1


async def test_action_unlikely_intent_2(
    monkeypatch: MonkeyPatch, tmp_path: Path, intent_ted_policy_moodbot_agent: Agent
):
    monkeypatch.setattr(
        SimplePolicyEnsemble,
        "probabilities_using_best_policy",
        _probabilities_with_action_unlikely_intent_for("mood_unhappy"),
    )

    file_name = tmp_path / "test_action_unlikely_intent_2.yml"
    file_name.write_text(
        """
        version: "2.0"
        stories:
          - story: unlikely path (with action_unlikely_intent)
            steps:
              - user: |
                  very terrible
                intent: mood_unhappy
              - action: action_unlikely_intent
              - action: utter_cheer_up
              - action: utter_did_that_help
              - intent: affirm
              - action: utter_happy
        """
    )

    result = await rasa.core.test.test(
        str(file_name), intent_ted_policy_moodbot_agent, out_directory=str(tmp_path),
    )
    assert "report" in result.keys()
    assert result["report"]["conversation_accuracy"]["correct"] == 1
    assert result["report"]["conversation_accuracy"]["with_warnings"] == 0


async def test_action_unlikely_intent_complete(
    monkeypatch: MonkeyPatch, tmp_path: Path, intent_ted_policy_moodbot_agent: Agent
):
    monkeypatch.setattr(
        SimplePolicyEnsemble,
        "probabilities_using_best_policy",
        _probabilities_with_action_unlikely_intent_for("mood_unhappy"),
    )

    file_name = tmp_path / "test_action_unlikely_intent_complete.yml"
    file_name.write_text(
        """
        version: "2.0"
        stories:
          - story: happy path
            steps:
              - user: |
                  hello there!
                intent: greet
              - action: utter_greet
              - user: |
                  amazing
                intent: mood_great
              - action: utter_happy
          - story: unlikely path
            steps:
              - user: |
                  very terrible
                intent: mood_unhappy
              - action: utter_cheer_up
              - action: utter_did_that_help
              - intent: affirm
              - action: utter_happy
          - story: unlikely path (with action_unlikely_intent)
            steps:
              - user: |
                  very terrible
                intent: mood_unhappy
              - action: action_unlikely_intent
              - action: utter_cheer_up
              - action: utter_did_that_help
              - intent: affirm
              - action: utter_happy
          - story: happy path 2
            steps:
              - user: |
                  hey!
                intent: greet
              - action: utter_greet
              - user: |
                  good
                intent: mood_great
              - action: utter_happy
        """
    )

    result = await rasa.core.test.test(
        str(file_name), intent_ted_policy_moodbot_agent, out_directory=str(tmp_path),
    )
    assert "report" in result.keys()
    assert result["report"]["conversation_accuracy"]["correct"] == 4
    assert result["report"]["conversation_accuracy"]["with_warnings"] == 1
    assert result["report"]["conversation_accuracy"]["total"] == 4


async def test_action_unlikely_intent_wrong_story(
    monkeypatch: MonkeyPatch, tmp_path: Path, intent_ted_policy_moodbot_agent: Agent
):
    monkeypatch.setattr(
        SimplePolicyEnsemble,
        "probabilities_using_best_policy",
        _probabilities_with_action_unlikely_intent_for("mood_unhappy"),
    )

    file_name = tmp_path / "test_action_unlikely_intent_complete.yml"
    file_name.write_text(
        """
        version: "2.0"
        stories:
          - story: happy path
            steps:
              - user: |
                  hello there!
                intent: greet
              - action: action_unlikely_intent
              - action: utter_greet
              - user: |
                  amazing
                intent: mood_great
              - action: utter_happy
        """
    )

    result = await rasa.core.test.test(
        str(file_name), intent_ted_policy_moodbot_agent, out_directory=str(tmp_path),
    )
    assert "report" in result.keys()
    assert result["report"]["conversation_accuracy"]["correct"] == 0
    assert result["report"]["conversation_accuracy"]["with_warnings"] == 0
    assert result["report"]["conversation_accuracy"]["total"] == 1
