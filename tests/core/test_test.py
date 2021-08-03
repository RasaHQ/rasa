from pathlib import Path
from typing import Text, Optional, Dict, Any, List, Callable
import pytest

import rasa.core.test
import rasa.shared.utils.io
from rasa.core.policies.ensemble import SimplePolicyEnsemble
from rasa.core.policies.policy import PolicyPrediction
from rasa.shared.core.events import UserUttered
from _pytest.monkeypatch import MonkeyPatch
from _pytest.capture import CaptureFixture
from rasa.core.agent import Agent
from rasa.utils.tensorflow.constants import (
    QUERY_INTENT_KEY,
    NAME,
    THRESHOLD_KEY,
    SEVERITY_KEY,
    SCORE_KEY,
)
from rasa.core.constants import STORIES_WITH_WARNINGS_FILE
from rasa.shared.core.constants import ACTION_UNLIKELY_INTENT_NAME
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.core.policies.rule_policy import RulePolicy
from rasa.shared.core.domain import State
from rasa.core.policies.policy import SupportedData


def _probabilities_with_action_unlikely_intent_for(
    intent_names: List[Text],
    metadata_for_intent: Optional[Dict[Text, Dict[Text, Any]]] = None,
) -> Callable[
    [SimplePolicyEnsemble, DialogueStateTracker, Domain, RegexInterpreter, Any],
    PolicyPrediction,
]:
    _original = SimplePolicyEnsemble.probabilities_using_best_policy

    def probabilities_using_best_policy(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: RegexInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:
        latest_event = tracker.events[-1]
        if (
            isinstance(latest_event, UserUttered)
            and latest_event.parse_data["intent"]["name"] in intent_names
        ):
            intent_name = latest_event.parse_data["intent"]["name"]
            # Here we return `action_unlikely_intent` if the name of the
            # latest intent is present in `intent_names`. Accompanying
            # metadata is fetched from `metadata_for_intent` if it is present.
            # We need to do it because every time the tests are run,
            # training will result in different model weights which might
            # result in different predictions of `action_unlikely_intent`.
            # Because we're not testing `UnexpecTEDIntentPolicy`,
            # here we simply trigger it by
            # predicting `action_unlikely_intent` in a specified moment
            # to make the tests deterministic.
            return PolicyPrediction.for_action_name(
                domain,
                ACTION_UNLIKELY_INTENT_NAME,
                action_metadata=metadata_for_intent.get(intent_name)
                if metadata_for_intent
                else None,
            )

        return _original(self, tracker, domain, interpreter, **kwargs)

    return probabilities_using_best_policy


def _custom_prediction_states_for_rules(
    ignore_action_unlikely_intent: bool = False,
) -> Callable[
    [RulePolicy, DialogueStateTracker, Domain, bool], List[State],
]:
    """Creates prediction states for `RulePolicy`.

    `RulePolicy` does not ignore `action_unlikely_intent` in reality.
    We use this helper method to monkey patch it for tests so that we can
    test `rasa test`'s behaviour when `action_unlikely_intent` is predicted.

    Args:
        ignore_action_unlikely_intent: Whether to ignore `action_unlikely_intent`.

    Returns:
        Monkey-patched method to create prediction states.
    """

    def _prediction_states(
        self: RulePolicy,
        tracker: DialogueStateTracker,
        domain: Domain,
        use_text_for_last_user_input: bool = False,
    ) -> List[State]:
        return self.featurizer.prediction_states(
            [tracker],
            domain,
            use_text_for_last_user_input=use_text_for_last_user_input,
            ignore_rule_only_turns=self.supported_data() == SupportedData.ML_DATA,
            rule_only_data=self._rule_only_data,
            ignore_action_unlikely_intent=ignore_action_unlikely_intent,
        )[0]

    return _prediction_states


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


def _train_rule_based_agent(
    moodbot_domain: Domain,
    train_file_name: Path,
    monkeypatch: MonkeyPatch,
    ignore_action_unlikely_intent: bool,
) -> Agent:

    # We need `RulePolicy` to predict the correct actions
    # in a particular conversation context as seen during training.
    # Since it can get affected by `action_unlikely_intent` being triggered in
    # some cases. We monkey-patch the method which creates
    # prediction states to ignore `action_unlikely_intent`s if needed.

    monkeypatch.setattr(
        RulePolicy,
        "_prediction_states",
        _custom_prediction_states_for_rules(ignore_action_unlikely_intent),
    )

    deterministic_policy = RulePolicy(restrict_rules=False)
    agent = Agent(moodbot_domain, SimplePolicyEnsemble([deterministic_policy]))
    training_data = agent.load_data(str(train_file_name))

    # Make the trackers compatible with rules
    # so that they are picked up by the policy.
    for tracker in training_data:
        tracker.is_rule_tracker = True

    agent.train(training_data)

    return agent


async def test_action_unlikely_intent_warning(
    monkeypatch: MonkeyPatch, tmp_path: Path, moodbot_domain: Domain
):
    monkeypatch.setattr(
        SimplePolicyEnsemble,
        "probabilities_using_best_policy",
        _probabilities_with_action_unlikely_intent_for(["mood_unhappy"]),
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

    # We train on the above story so that RulePolicy can memorize
    # it and we don't have to worry about other actions being
    # predicted correctly.
    agent = await _train_rule_based_agent(
        moodbot_domain, file_name, monkeypatch, ignore_action_unlikely_intent=True
    )

    result = await rasa.core.test.test(
        str(file_name),
        agent,
        out_directory=str(tmp_path),
        fail_on_prediction_errors=True,
    )
    assert "report" in result.keys()
    assert result["report"]["conversation_accuracy"]["correct"] == 1
    assert result["report"]["conversation_accuracy"]["with_warnings"] == 1

    # Ensure that the story with warning is correctly formatted
    with open(str(tmp_path / "stories_with_warnings.yml"), "r") as f:
        content = f.read()
        assert f"# predicted: {ACTION_UNLIKELY_INTENT_NAME}" in content


async def test_action_unlikely_intent_correctly_predicted(
    monkeypatch: MonkeyPatch, tmp_path: Path, moodbot_domain: Domain
):
    monkeypatch.setattr(
        SimplePolicyEnsemble,
        "probabilities_using_best_policy",
        _probabilities_with_action_unlikely_intent_for(["mood_unhappy"]),
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

    # We train on the above story so that RulePolicy can memorize
    # it and we don't have to worry about other actions being
    # predicted correctly.
    agent = await _train_rule_based_agent(
        moodbot_domain, file_name, monkeypatch, ignore_action_unlikely_intent=False
    )

    result = await rasa.core.test.test(
        str(file_name),
        agent,
        out_directory=str(tmp_path),
        fail_on_prediction_errors=True,
    )
    assert "report" in result.keys()
    assert result["report"]["conversation_accuracy"]["correct"] == 1
    assert result["report"]["conversation_accuracy"]["with_warnings"] == 0


async def test_wrong_action_after_action_unlikely_intent(
    monkeypatch: MonkeyPatch, tmp_path: Path, moodbot_domain: Domain
):
    monkeypatch.setattr(
        SimplePolicyEnsemble,
        "probabilities_using_best_policy",
        _probabilities_with_action_unlikely_intent_for(["greet", "mood_great"]),
    )

    test_file_name = tmp_path / "test.yml"
    test_file_name.write_text(
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
        """
    )

    train_file_name = tmp_path / "train.yml"
    train_file_name.write_text(
        """
        version: "2.0"
        stories:
          - story: happy path
            steps:
              - user: |
                  hello there!
                intent: greet
              - action: utter_happy
              - user: |
                  amazing
                intent: mood_great
              - action: utter_goodbye
        """
    )

    # We train on the above story so that RulePolicy can memorize
    # it and we don't have to worry about other actions being
    # predicted correctly.
    agent = await _train_rule_based_agent(
        moodbot_domain, train_file_name, monkeypatch, ignore_action_unlikely_intent=True
    )

    result = await rasa.core.test.test(
        str(test_file_name),
        agent,
        out_directory=str(tmp_path),
        fail_on_prediction_errors=False,
    )
    assert "report" in result.keys()
    assert result["report"]["conversation_accuracy"]["correct"] == 0
    assert result["report"]["conversation_accuracy"]["with_warnings"] == 0
    assert result["report"]["conversation_accuracy"]["total"] == 1

    # Ensure that the failed story is correctly formatted
    with open(str(tmp_path / "failed_test_stories.yml"), "r") as f:
        content = f.read()
        assert (
            f"# predicted: utter_happy after {ACTION_UNLIKELY_INTENT_NAME}" in content
        )
        assert (
            f"# predicted: action_default_fallback after {ACTION_UNLIKELY_INTENT_NAME}"
            in content
        )


async def test_action_unlikely_intent_not_found(
    monkeypatch: MonkeyPatch, tmp_path: Path, moodbot_domain: Domain
):
    test_file_name = tmp_path / "test_action_unlikely_intent_complete.yml"
    test_file_name.write_text(
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

    train_file_name = tmp_path / "train_without_action_unlikely_intent.yml"
    train_file_name.write_text(
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
        """
    )

    # We train on the above story so that RulePolicy can memorize
    # it and we don't have to worry about other actions being
    # predicted correctly.
    agent = await _train_rule_based_agent(
        moodbot_domain,
        train_file_name,
        monkeypatch,
        ignore_action_unlikely_intent=False,
    )

    result = await rasa.core.test.test(
        str(test_file_name), agent, out_directory=str(tmp_path)
    )
    assert "report" in result.keys()
    assert result["report"]["conversation_accuracy"]["correct"] == 0
    assert result["report"]["conversation_accuracy"]["with_warnings"] == 0
    assert result["report"]["conversation_accuracy"]["total"] == 1

    # Ensure that the failed story is correctly formatted
    with open(str(tmp_path / "failed_test_stories.yml"), "r") as f:
        content = f.read()
        assert "# predicted: utter_greet" in content


async def test_action_unlikely_intent_warning_and_story_error(
    monkeypatch: MonkeyPatch, tmp_path: Path, moodbot_domain: Domain
):
    monkeypatch.setattr(
        SimplePolicyEnsemble,
        "probabilities_using_best_policy",
        _probabilities_with_action_unlikely_intent_for(["greet"]),
    )

    test_file_name = tmp_path / "test.yml"
    test_file_name.write_text(
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
        """
    )

    train_file_name = tmp_path / "train.yml"
    train_file_name.write_text(
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
              - action: utter_goodbye
        """
    )

    # We train on the above story so that RulePolicy can memorize
    # it and we don't have to worry about other actions being
    # predicted correctly.
    agent = await _train_rule_based_agent(
        moodbot_domain, train_file_name, monkeypatch, ignore_action_unlikely_intent=True
    )

    result = await rasa.core.test.test(
        str(test_file_name), agent, out_directory=str(tmp_path),
    )
    assert "report" in result.keys()
    assert result["report"]["conversation_accuracy"]["correct"] == 0
    assert result["report"]["conversation_accuracy"]["with_warnings"] == 0
    assert result["report"]["conversation_accuracy"]["total"] == 1

    # Ensure that the failed story is correctly formatted
    with open(str(tmp_path / "failed_test_stories.yml"), "r") as f:
        content = f.read()
        assert f"# predicted: {ACTION_UNLIKELY_INTENT_NAME}" in content
        assert "# predicted: utter_goodbye" in content


async def test_fail_on_prediction_errors(
    monkeypatch: MonkeyPatch, tmp_path: Path, moodbot_domain: Domain
):
    monkeypatch.setattr(
        SimplePolicyEnsemble,
        "probabilities_using_best_policy",
        _probabilities_with_action_unlikely_intent_for(["mood_unhappy"]),
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
              - action: utter_cheer_up
              - action: action_unlikely_intent
              - action: utter_did_that_help
              - intent: affirm
              - action: utter_happy
        """
    )

    # We train on the above story so that RulePolicy can memorize
    # it and we don't have to worry about other actions being
    # predicted correctly.
    agent = await _train_rule_based_agent(
        moodbot_domain, file_name, monkeypatch, ignore_action_unlikely_intent=False
    )

    with pytest.raises(rasa.core.test.WrongPredictionException):
        await rasa.core.test.test(
            str(file_name),
            agent,
            out_directory=str(tmp_path),
            fail_on_prediction_errors=True,
        )


@pytest.mark.parametrize(
    "metadata_for_intents, story_order",
    [
        (
            {
                "mood_unhappy": {
                    QUERY_INTENT_KEY: {
                        NAME: "mood_unhappy",
                        SEVERITY_KEY: 2.0,
                        THRESHOLD_KEY: 0.0,
                        SCORE_KEY: -2.0,
                    }
                },
                "mood_great": {
                    QUERY_INTENT_KEY: {
                        NAME: "mood_great",
                        SEVERITY_KEY: 3.0,
                        THRESHOLD_KEY: 0.2,
                        SCORE_KEY: -1.0,
                    }
                },
                "affirm": {
                    QUERY_INTENT_KEY: {
                        NAME: "affirm",
                        SEVERITY_KEY: 4.2,
                        THRESHOLD_KEY: 0.2,
                        SCORE_KEY: -4.0,
                    }
                },
            },
            ["path 2", "path 1"],
        ),
        (
            {
                "mood_unhappy": {
                    QUERY_INTENT_KEY: {
                        NAME: "mood_unhappy",
                        SEVERITY_KEY: 2.0,
                        THRESHOLD_KEY: 0.0,
                        SCORE_KEY: -2.0,
                    }
                },
                "mood_great": {
                    QUERY_INTENT_KEY: {
                        NAME: "mood_great",
                        SEVERITY_KEY: 5.0,
                        THRESHOLD_KEY: 0.2,
                        SCORE_KEY: -1.0,
                    }
                },
                "affirm": {
                    QUERY_INTENT_KEY: {
                        NAME: "affirm",
                        SEVERITY_KEY: 4.2,
                        THRESHOLD_KEY: 0.2,
                        SCORE_KEY: -4.0,
                    }
                },
            },
            ["path 1", "path 2"],
        ),
    ],
)
async def test_multiple_warnings_sorted_on_severity(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    moodbot_domain: Domain,
    metadata_for_intents: Dict,
    story_order: List[Text],
):
    monkeypatch.setattr(
        SimplePolicyEnsemble,
        "probabilities_using_best_policy",
        _probabilities_with_action_unlikely_intent_for(
            list(metadata_for_intents.keys()), metadata_for_intents
        ),
    )

    test_story_path = (
        "data/test_yaml_stories/test_multiple_action_unlikely_intent_warnings.yml"
    )

    # We train on the stories as it is so that RulePolicy can memorize
    # it and we don't have to worry about other actions being
    # predicted correctly.
    agent = await _train_rule_based_agent(
        moodbot_domain,
        Path(test_story_path),
        monkeypatch,
        ignore_action_unlikely_intent=True,
    )

    await rasa.core.test.test(
        test_story_path,
        agent,
        out_directory=str(tmp_path),
        fail_on_prediction_errors=True,
    )

    warnings_file = tmp_path / STORIES_WITH_WARNINGS_FILE
    warnings_data = rasa.shared.utils.io.read_yaml_file(warnings_file)

    for index, story_name in enumerate(story_order):
        assert warnings_data["stories"][index]["story"].startswith(story_name)
