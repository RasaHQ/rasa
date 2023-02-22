import shutil
import textwrap
from pathlib import Path
from typing import Text, Optional, Dict, Any, List, Callable, Coroutine
import pytest
import rasa.core.test
import rasa.shared.utils.io
from rasa.core.policies.ensemble import DefaultPolicyPredictionEnsemble
from rasa.core.policies.policy import PolicyPrediction
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.events import UserUttered
from _pytest.monkeypatch import MonkeyPatch
from _pytest.capture import CaptureFixture
from rasa.core.agent import Agent, load_agent
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

from rasa.core.policies.rule_policy import RulePolicy
from rasa.shared.core.domain import State
from rasa.core.policies.policy import SupportedData
from rasa.shared.utils.io import read_file, read_yaml


def _probabilities_with_action_unlikely_intent_for(
    intent_names: List[Text],
    metadata_for_intent: Optional[Dict[Text, Dict[Text, Any]]] = None,
) -> Callable[
    [DefaultPolicyPredictionEnsemble, DialogueStateTracker, Domain, Any],
    PolicyPrediction,
]:
    _original = DefaultPolicyPredictionEnsemble.combine_predictions_from_kwargs

    def combine_predictions_from_kwargs(
        self, tracker: DialogueStateTracker, domain: Domain, **kwargs: Any
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

        return _original(self, tracker, domain, **kwargs)

    return combine_predictions_from_kwargs


def _custom_prediction_states_for_rules(
    ignore_action_unlikely_intent: bool = False,
) -> Callable[[RulePolicy, DialogueStateTracker, Domain, bool], List[State],]:
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
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[State]:
        return self.featurizer.prediction_states(
            [tracker],
            domain,
            use_text_for_last_user_input=use_text_for_last_user_input,
            ignore_rule_only_turns=self.supported_data() == SupportedData.ML_DATA,
            rule_only_data=rule_only_data,
            ignore_action_unlikely_intent=ignore_action_unlikely_intent,
        )[0]

    return _prediction_states


# FIXME: these tests take too long to run in the CI, disabling them for now
@pytest.mark.skip_on_ci
async def test_testing_warns_if_action_unknown(
    capsys: CaptureFixture,
    e2e_bot_agent: Agent,
    e2e_bot_test_stories_with_unknown_bot_utterances: Path,
    tmp_path: Path,
):
    await rasa.core.test.test(
        e2e_bot_test_stories_with_unknown_bot_utterances,
        e2e_bot_agent,
        out_directory=str(tmp_path),
    )
    output = capsys.readouterr().out
    assert "Test story" in output
    assert "contains the bot utterance" in output
    assert "which is not part of the training data / domain" in output


async def test_testing_with_utilizing_retrieval_intents(
    response_selector_agent: Agent, response_selector_test_stories: Path, tmp_path: Path
):
    result = await rasa.core.test.test(
        stories=response_selector_test_stories,
        agent=response_selector_agent,
        e2e=True,
        out_directory=str(tmp_path),
        disable_plotting=True,
        warnings=False,
    )
    failed_stories_path = tmp_path / "failed_test_stories.yml"
    failed_stories = read_yaml(read_file(failed_stories_path, "utf-8"))
    # check that the intent is shown correctly in the failed test stories file
    target_intents = {
        "test 0": "chitchat/ask_name",
        "test 1": "chitchat/ask_name",
        "test 2": "chitchat",
        "test 3": "chitchat",
    }
    for story in failed_stories["stories"]:
        test_name = story["story"].split("-")[0].strip()
        assert story["steps"][0]["intent"] == target_intents[test_name]
    # check that retrieval intent for actions is retrieved correctly
    # and only when it's needed.
    target_actions = {
        "utter_chitchat": "utter_chitchat",
        "utter_chitchat/ask_name": "utter_chitchat/ask_name",
        "utter_chitchat/ask_weather": "utter_chitchat/ask_name",
        "utter_goodbye": "utter_chitchat/ask_name",
    }
    predicted_actions = result["actions"][::2]
    for predicted_action in predicted_actions:
        assert (
            target_actions[predicted_action["action"]] == predicted_action["predicted"]
        )


async def test_testing_does_not_warn_if_intent_in_domain(
    default_agent: Agent, stories_path: Text, tmp_path: Path
):
    with pytest.warns(UserWarning) as record:
        await rasa.core.test.test(
            Path(stories_path), default_agent, out_directory=str(tmp_path)
        )

    assert not any("Found intent" in r.message.args[0] for r in record)
    assert all(
        "in stories which is not part of the domain" not in r.message.args[0]
        for r in record
    )


async def test_testing_valid_with_non_e2e_core_model(core_agent: Agent, tmp_path: Path):
    result = await rasa.core.test.test(
        "data/test_yaml_stories/test_stories_entity_annotations.yml",
        core_agent,
        out_directory=str(tmp_path),
    )
    assert "report" in result.keys()


@pytest.fixture()
async def _train_rule_based_agent(
    moodbot_domain: Domain,
    tmp_path: Path,
    trained_async: Callable,
    monkeypatch: MonkeyPatch,
    moodbot_domain_path: Path,
) -> Callable[[Path, bool], Coroutine]:

    # We need `RulePolicy` to predict the correct actions
    # in a particular conversation context as seen during training.
    # Since it can get affected by `action_unlikely_intent` being triggered in
    # some cases. We monkey-patch the method which creates
    # prediction states to ignore `action_unlikely_intent`s if needed.

    async def inner(file_name: Path, ignore_action_unlikely_intent: bool) -> Agent:
        config = textwrap.dedent(
            f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        assistant_id: placeholder_default
        pipeline: []
        policies:
        - name: RulePolicy
          restrict_rules: false
        """
        )
        config_path = tmp_path / "config.yml"
        rasa.shared.utils.io.write_text_file(config, config_path)

        rule_file = tmp_path / "rules.yml"
        shutil.copy2(file_name, rule_file)
        training_data = rule_file.read_text()
        training_data_for_rules = training_data.replace("stories:", "rules:")
        training_data_for_rules = training_data_for_rules.replace("story:", "rule:")
        rule_file.write_text(training_data_for_rules)

        model_path = await trained_async(
            moodbot_domain_path, str(config_path), str(rule_file)
        )

        monkeypatch.setattr(
            RulePolicy,
            "_prediction_states",
            _custom_prediction_states_for_rules(ignore_action_unlikely_intent),
        )

        return await load_agent(model_path)

    return inner


async def test_action_unlikely_intent_warning(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    _train_rule_based_agent: Callable[[Path, bool], Coroutine],
):
    monkeypatch.setattr(
        DefaultPolicyPredictionEnsemble,
        "combine_predictions_from_kwargs",
        _probabilities_with_action_unlikely_intent_for(["mood_unhappy"]),
    )

    file_name = tmp_path / "test_action_unlikely_intent_1.yml"
    file_name.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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
    agent = await _train_rule_based_agent(file_name, True)

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
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    _train_rule_based_agent: Callable[[Path, bool], Coroutine],
):
    monkeypatch.setattr(
        DefaultPolicyPredictionEnsemble,
        "combine_predictions_from_kwargs",
        _probabilities_with_action_unlikely_intent_for(["mood_unhappy"]),
    )

    file_name = tmp_path / "test_action_unlikely_intent_2.yml"
    file_name.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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
    agent = await _train_rule_based_agent(file_name, False)

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
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    _train_rule_based_agent: Callable[[Path, bool], Coroutine],
):
    monkeypatch.setattr(
        DefaultPolicyPredictionEnsemble,
        "combine_predictions_from_kwargs",
        _probabilities_with_action_unlikely_intent_for(["greet", "mood_great"]),
    )

    test_file_name = tmp_path / "test.yml"
    test_file_name.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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
    agent = await _train_rule_based_agent(train_file_name, True)

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
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    _train_rule_based_agent: Callable[[Path, bool], Coroutine],
):
    test_file_name = tmp_path / "test_action_unlikely_intent_complete.yml"
    test_file_name.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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
    agent = await _train_rule_based_agent(train_file_name, False)

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
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    _train_rule_based_agent: Callable[[Path, bool], Coroutine],
):
    monkeypatch.setattr(
        DefaultPolicyPredictionEnsemble,
        "combine_predictions_from_kwargs",
        _probabilities_with_action_unlikely_intent_for(["greet"]),
    )

    test_file_name = tmp_path / "test.yml"
    test_file_name.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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
    agent = await _train_rule_based_agent(train_file_name, True)

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
        assert f"# predicted: {ACTION_UNLIKELY_INTENT_NAME}" in content
        assert "# predicted: utter_goodbye" in content


async def test_fail_on_prediction_errors(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    _train_rule_based_agent: Callable[[Path, bool], Coroutine],
):
    monkeypatch.setattr(
        DefaultPolicyPredictionEnsemble,
        "combine_predictions_from_kwargs",
        _probabilities_with_action_unlikely_intent_for(["mood_unhappy"]),
    )

    file_name = tmp_path / "test_action_unlikely_intent_2.yml"
    file_name.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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
    agent = await _train_rule_based_agent(file_name, False)

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
    _train_rule_based_agent: Callable[[Path, bool], Coroutine],
    metadata_for_intents: Dict,
    story_order: List[Text],
):
    monkeypatch.setattr(
        DefaultPolicyPredictionEnsemble,
        "combine_predictions_from_kwargs",
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
    agent = await _train_rule_based_agent(Path(test_story_path), True)

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
