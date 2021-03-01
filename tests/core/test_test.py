import asyncio
import json
import pathlib
import pytest
from typing import Text, Any, Dict

from rasa.core.agent import Agent
from rasa.core.policies.ensemble import SimplePolicyEnsemble
from rasa.core.policies.rule_policy import RulePolicy
import rasa.core.test
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.nlu.interpreter import RegexInterpreter


@pytest.fixture(scope="session")
def out_directory(tmpdir_factory):
    """Output directory for logging info."""
    fn = tmpdir_factory.mktemp("results")
    return fn


@pytest.mark.parametrize(
    "stories_yaml,expected_results",
    [
        [
            """
stories:
  - story: story1
    steps:
    - intent: intentA
    - action: actionA
  - story: story2
    steps:
    - intent: intentB
    - action: actionB
  - story: story3
    steps:
    - intent: intentA
    - action: actionA
    - intent: intentB
    - action: actionC
            """,
            {
                "actionB": {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1-score": 1.0,
                    "support": 1,
                },
                "action_listen": {
                    "precision": 1.0,
                    "recall": 0.75,
                    "f1-score": 0.8571428571428571,
                    "support": 4,
                },
                "actionA": {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1-score": 1.0,
                    "support": 2,
                },
                "actionC": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1-score": 0.0,
                    "support": 1,
                },
                "micro avg": {
                    "precision": 1.0,
                    "recall": 0.75,
                    "f1-score": 0.8571428571428571,
                    "support": 8,
                },
                "macro avg": {
                    "precision": 0.75,
                    "recall": 0.6875,
                    "f1-score": 0.7142857142857143,
                    "support": 8,
                },
                "weighted avg": {
                    "precision": 0.875,
                    "recall": 0.75,
                    "f1-score": 0.8035714285714286,
                    "support": 8,
                },
                "conversation_accuracy": {
                    "accuracy": 2.0 / 3.0,
                    "total": 3,
                    "correct": 2,
                },
            },
        ],
        ["", {}],
    ],
)
async def test_test(
    tmpdir_factory: pathlib.Path,
    out_directory: pathlib.Path,
    stories_yaml: Text,
    expected_results: Dict[Text, Dict[Text, Any]],
) -> None:

    stories_path = tmpdir_factory.mktemp("test_rasa_core_test").join("eval_stories.yml")
    stories_path.write_text(stories_yaml, "utf8")

    domain = Domain.from_yaml(
        """
intents:
- intentA
- intentB
actions:
- actionA
- actionB
- actionC
"""
    )

    policy = RulePolicy()
    rt1 = TrackerWithCachedStates.from_events(
        "ruleAtoA",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": "intentA"}),
            ActionExecuted("actionA"),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )
    rt2 = TrackerWithCachedStates.from_events(
        "ruleBtoB",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": "intentB"}),
            ActionExecuted("actionB"),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )

    policy.train([rt1, rt2], domain, RegexInterpreter())

    agent = Agent(domain=domain, policies=SimplePolicyEnsemble([policy]),)

    await rasa.core.test.test(stories_path, agent, out_directory=out_directory)
    story_report_path = out_directory / "story_report.json"
    assert story_report_path.exists()

    actual_results = json.loads(story_report_path.read_text("utf8"))
    assert actual_results == expected_results
