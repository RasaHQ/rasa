from pathlib import Path

from rasa.core.agent import Agent
import rasa.model_training
import rasa.shared.utils.io
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION


async def test_rule_policy_initial_slot_values_ignored_in_rules(tmp_path: Path):
    """
    Tests that intial slot values don't affect rules when full train-test pipeline runs.

    This test case was introduced as an extension of previous ones which did not run
    the full train-test pipeline and hence failed to catch a bug.
    """
    policies_config = {"policies": [{"name": "RulePolicy"}], "recipe": "default.v1"}
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(policies_config, config_file)

    domain = f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
          - greet
          - goodbye
        slots:
          slot1:
            type: categorical
            values:
              - default_value
              - non_default_value
            initial_value: default_value
            mappings:
            - type: from_entity
              entity: entity1
        responses:
          utter_goodbye:
          - text: "Bye"
    """
    domain_path = tmp_path / "domain.yml"
    domain_path.write_text(domain)

    # this rule does not mention any slots and so should apply regardless of slot values
    train_rule = f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        rules:
        - rule: training rule
          steps:
          - intent: goodbye
          - action: utter_goodbye
    """
    train_rule_path = tmp_path / "train_story.yml"
    train_rule_path.write_text(train_rule)

    # this story features a slot set to its non-initial value but the learned rule
    # should still be applicable and predict `utter_goodbye`
    test_story = f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        stories:
        - story: test story
          steps:
          - slot_was_set:
            - slot1: non_default_value
          - intent: goodbye
          - action: utter_goodbye
    """
    test_story_path = tmp_path / "test_story.yml"
    test_story_path.write_text(test_story)

    model_file = rasa.model_training.train_core(
        domain_path,
        config=str(config_file),
        stories=train_rule_path,
        output=str(tmp_path),
    )
    agent = Agent.load(model_file)
    result = await rasa.core.test.test(
        str(test_story_path),
        agent,
        out_directory=str(tmp_path),
        fail_on_prediction_errors=True,
    )

    assert (
        result.get("report", {}).get("conversation_accuracy", {}).get("correct", None)
        == 1
    )
