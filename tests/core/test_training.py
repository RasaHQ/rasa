from pathlib import Path
from unittest.mock import Mock
from _pytest.monkeypatch import MonkeyPatch
from typing import Text

import pytest

from rasa.core import training
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.constants import DEFAULT_MAX_HISTORY
from rasa.core.policies.rule_policy import RulePolicy
from rasa.shared.core.domain import Domain
from rasa.core.interpreter import RasaNLUInterpreter
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.core.train import train
from rasa.core.agent import Agent
from rasa.core.policies.ted_policy import TEDPolicy

from rasa.shared.core.training_data.visualization import visualize_stories


def test_load_training_data_reader_not_found_throws(tmp_path: Path, domain: Domain):
    (tmp_path / "file").touch()

    with pytest.raises(Exception):
        training.load_data(str(tmp_path), domain)


async def test_story_visualization(domain: Domain, tmp_path: Path):
    import rasa.shared.core.training_data.loading as core_loading

    story_steps = core_loading.load_data_from_resource(
        "data/test_yaml_stories/stories.yml", domain
    )
    out_file = str(tmp_path / "graph.html")
    generated_graph = await visualize_stories(
        story_steps,
        domain,
        output_file=out_file,
        max_history=3,
        should_merge_nodes=False,
    )

    assert len(generated_graph.nodes()) == 51

    assert len(generated_graph.edges()) == 56


async def test_story_visualization_with_merging(domain: Domain):
    import rasa.shared.core.training_data.loading as core_loading

    story_steps = core_loading.load_data_from_resource(
        "data/test_yaml_stories/stories.yml", domain
    )
    generated_graph = await visualize_stories(
        story_steps, domain, output_file=None, max_history=3, should_merge_nodes=True,
    )
    assert 15 < len(generated_graph.nodes()) < 33

    assert 20 < len(generated_graph.edges()) < 33


def test_training_script_without_max_history_set(
    tmp_path: Path, domain_path: Text, stories_path: Text
):
    tmpdir = str(tmp_path)

    train(
        domain_path,
        stories_path,
        tmpdir,
        interpreter=RegexInterpreter(),
        policy_config="data/test_config/no_max_hist_config.yml",
        additional_arguments={},
    )

    agent = Agent.load(tmpdir)
    for policy in agent.policy_ensemble.policies:
        if hasattr(policy.featurizer, "max_history"):
            if type(policy) == MemoizationPolicy:
                assert policy.featurizer.max_history == DEFAULT_MAX_HISTORY
            else:
                assert policy.featurizer.max_history is None


def test_training_script_with_max_history_set(
    tmp_path: Path, domain_path: Text, stories_path: Text
):
    tmpdir = str(tmp_path)

    train(
        domain_path,
        stories_path,
        tmpdir,
        interpreter=RegexInterpreter(),
        policy_config="data/test_config/max_hist_config.yml",
        additional_arguments={},
    )
    agent = Agent.load(tmpdir)

    expected_max_history = {RulePolicy: None}
    for policy in agent.policy_ensemble.policies:
        if hasattr(policy.featurizer, "max_history"):
            expected_history = expected_max_history.get(type(policy), 5)
            assert policy.featurizer.max_history == expected_history


def test_training_script_with_restart_stories(tmp_path: Path, domain_path: Text):
    train(
        domain_path,
        "data/test_yaml_stories/stories_restart.yml",
        str(tmp_path),
        interpreter=RegexInterpreter(),
        policy_config="data/test_config/max_hist_config.yml",
        additional_arguments={},
    )
    assert True


@pytest.mark.timeout(120, func_only=True)
async def test_random_seed(
    tmp_path: Path, monkeypatch: MonkeyPatch, domain_path: Text, stories_path: Text
):
    policies_config = {
        "policies": [
            {"name": TEDPolicy.__name__, "random_seed": 42},
            {"name": RulePolicy.__name__},
        ]
    }

    agent_1 = train(
        domain_path,
        stories_path,
        str(tmp_path),
        interpreter=RegexInterpreter(),
        policy_config=policies_config,
        additional_arguments={},
    )

    agent_2 = train(
        domain_path,
        stories_path,
        str(tmp_path),
        interpreter=RegexInterpreter(),
        policy_config=policies_config,
        additional_arguments={},
    )

    processor_1 = agent_1.create_processor()
    processor_2 = agent_2.create_processor()

    probs_1 = await processor_1.predict_next("1")
    probs_2 = await processor_2.predict_next("2")
    assert probs_1["confidence"] == probs_2["confidence"]


def test_trained_interpreter_passed_to_policies(
    tmp_path: Path, monkeypatch: MonkeyPatch, domain_path: Text, stories_path: Text
):
    policies_config = {
        "policies": [{"name": TEDPolicy.__name__}, {"name": RulePolicy.__name__}]
    }

    policy_train = Mock()
    monkeypatch.setattr(TEDPolicy, "train", policy_train)

    interpreter = Mock(spec=RasaNLUInterpreter)

    train(
        domain_path,
        stories_path,
        str(tmp_path),
        interpreter=interpreter,
        policy_config=policies_config,
        additional_arguments={},
    )

    policy_train.assert_called_once()

    assert policy_train.call_count == 1
    _, _, kwargs = policy_train.mock_calls[0]
    assert kwargs["interpreter"] == interpreter
