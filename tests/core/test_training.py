from pathlib import Path
from typing import List, Text
from unittest.mock import Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.core.policies.memoization import MemoizationPolicy, OLD_DEFAULT_MAX_HISTORY
from rasa.core.policies.rule_policy import RulePolicy
from rasa.shared.core.domain import Domain
from rasa.core.interpreter import RasaNLUInterpreter
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.core.train import train
from rasa.core.agent import Agent
from rasa.core.policies.form_policy import FormPolicy

from rasa.shared.core.training_data.visualization import visualize_stories


async def test_story_visualization(domain: Domain, tmp_path: Path):
    import rasa.shared.core.training_data.loading as core_loading

    story_steps = await core_loading.load_data_from_resource(
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

    story_steps = await core_loading.load_data_from_resource(
        "data/test_yaml_stories/stories.yml", domain
    )
    generated_graph = await visualize_stories(
        story_steps, domain, output_file=None, max_history=3, should_merge_nodes=True,
    )
    assert 15 < len(generated_graph.nodes()) < 33

    assert 20 < len(generated_graph.edges()) < 33


async def test_training_script(tmp_path: Path, domain_path: Text, stories_path: Text):
    await train(
        domain_path,
        stories_path,
        str(tmp_path),
        policy_config="data/test_config/max_hist_config.yml",
        interpreter=RegexInterpreter(),
        additional_arguments={},
    )
    assert True


async def test_training_script_without_max_history_set(
    tmp_path: Path, domain_path: Text, stories_path: Text
):
    tmpdir = str(tmp_path)
    await train(
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
            if type(policy) == FormPolicy:
                assert policy.featurizer.max_history == 2
            elif type(policy) == MemoizationPolicy:
                assert policy.featurizer.max_history == OLD_DEFAULT_MAX_HISTORY
            else:
                assert policy.featurizer.max_history is None


async def test_training_script_with_max_history_set(
    tmp_path: Path, domain_path: Text, stories_path: Text
):
    tmpdir = str(tmp_path)

    await train(
        domain_path,
        stories_path,
        tmpdir,
        interpreter=RegexInterpreter(),
        policy_config="data/test_config/max_hist_config.yml",
        additional_arguments={},
    )
    agent = Agent.load(tmpdir)

    expected_max_history = {FormPolicy: 2, RulePolicy: None}
    for policy in agent.policy_ensemble.policies:
        if hasattr(policy.featurizer, "max_history"):
            expected_history = expected_max_history.get(type(policy), 5)
            assert policy.featurizer.max_history == expected_history


async def test_training_script_with_restart_stories(tmp_path: Path, domain_path: Text):
    await train(
        domain_path,
        "data/test_yaml_stories/stories_restart.yml",
        str(tmp_path),
        interpreter=RegexInterpreter(),
        policy_config="data/test_config/max_hist_config.yml",
        additional_arguments={},
    )
    assert True


def configs_for_random_seed_test() -> List[Text]:
    # define the configs for the random_seed tests
    return ["data/test_config/ted_random_seed.yaml"]


@pytest.mark.parametrize("config_file", configs_for_random_seed_test())
async def test_random_seed(
    tmp_path: Path, config_file: Text, domain_path: Text, stories_path: Text
):
    # set random seed in config file to
    # generate a reproducible training result

    agent_1 = await train(
        domain_path,
        stories_path,
        str(tmp_path / "1"),
        interpreter=RegexInterpreter(),
        policy_config=config_file,
        additional_arguments={},
    )

    agent_2 = await train(
        domain_path,
        stories_path,
        str(tmp_path / "2"),
        interpreter=RegexInterpreter(),
        policy_config=config_file,
        additional_arguments={},
    )

    processor_1 = agent_1.create_processor()
    processor_2 = agent_2.create_processor()

    probs_1 = await processor_1.predict_next("1")
    probs_2 = await processor_2.predict_next("2")
    assert probs_1["confidence"] == probs_2["confidence"]


async def test_trained_interpreter_passed_to_policies(
    tmp_path: Path, monkeypatch: MonkeyPatch, domain_path: Text, stories_path: Text
):
    from rasa.core.policies.ted_policy import TEDPolicy

    policies_config = {
        "policies": [{"name": TEDPolicy.__name__}, {"name": RulePolicy.__name__}]
    }

    policy_train = Mock()
    monkeypatch.setattr(TEDPolicy, "train", policy_train)

    interpreter = Mock(spec=RasaNLUInterpreter)

    await train(
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
