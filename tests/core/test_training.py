from pathlib import Path
from typing import List, Text
from unittest.mock import Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.core.domain import Domain
from rasa.core.interpreter import RegexInterpreter, RasaNLUInterpreter
from rasa.core.train import train
from rasa.core.agent import Agent
from rasa.core.policies.form_policy import FormPolicy

from rasa.core.training.visualization import visualize_stories
from tests.core.conftest import DEFAULT_DOMAIN_PATH_WITH_SLOTS, DEFAULT_STORIES_FILE


@pytest.mark.parametrize(
    "stories_file",
    ["data/test_stories/stories.md", "data/test_yaml_stories/stories.yml"],
)
async def test_story_visualization(
    stories_file: Text, default_domain: Domain, tmp_path: Path
):
    import rasa.core.training.loading as core_loading

    story_steps = await core_loading.load_data_from_resource(
        "data/test_stories/stories.md", default_domain
    )
    out_file = str(tmp_path / "graph.html")
    generated_graph = await visualize_stories(
        story_steps,
        default_domain,
        output_file=out_file,
        max_history=3,
        should_merge_nodes=False,
    )

    assert len(generated_graph.nodes()) == 51

    assert len(generated_graph.edges()) == 56


@pytest.mark.parametrize(
    "stories_file",
    ["data/test_stories/stories.md", "data/test_yaml_stories/stories.yml"],
)
async def test_story_visualization_with_merging(
    stories_file: Text, default_domain: Domain
):
    import rasa.core.training.loading as core_loading

    story_steps = await core_loading.load_data_from_resource(
        stories_file, default_domain
    )
    generated_graph = await visualize_stories(
        story_steps,
        default_domain,
        output_file=None,
        max_history=3,
        should_merge_nodes=True,
    )
    assert 15 < len(generated_graph.nodes()) < 33

    assert 20 < len(generated_graph.edges()) < 33


async def test_training_script(tmp_path: Path):
    await train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        DEFAULT_STORIES_FILE,
        str(tmp_path),
        policy_config="data/test_config/max_hist_config.yml",
        interpreter=RegexInterpreter(),
        additional_arguments={},
    )
    assert True


async def test_training_script_without_max_history_set(tmp_path: Path):
    tmpdir = str(tmp_path)
    await train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        DEFAULT_STORIES_FILE,
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
            else:
                assert (
                    policy.featurizer.max_history
                    == policy.featurizer.MAX_HISTORY_DEFAULT
                )


async def test_training_script_with_max_history_set(tmp_path: Path):
    tmpdir = str(tmp_path)

    await train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        DEFAULT_STORIES_FILE,
        tmpdir,
        interpreter=RegexInterpreter(),
        policy_config="data/test_config/max_hist_config.yml",
        additional_arguments={},
    )
    agent = Agent.load(tmpdir)
    for policy in agent.policy_ensemble.policies:
        if hasattr(policy.featurizer, "max_history"):
            if type(policy) == FormPolicy:
                assert policy.featurizer.max_history == 2
            else:
                assert policy.featurizer.max_history == 5


@pytest.mark.parametrize(
    "stories_file",
    [
        "data/test_stories/stories_restart.md",
        "data/test_yaml_stories/stories_restart.yml",
    ],
)
async def test_training_script_with_restart_stories(stories_file: Text, tmp_path: Path):
    await train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        stories_file,
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
async def test_random_seed(tmp_path: Path, config_file: Text):
    # set random seed in config file to
    # generate a reproducible training result

    agent_1 = await train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        DEFAULT_STORIES_FILE,
        str(tmp_path / "1"),
        interpreter=RegexInterpreter(),
        policy_config=config_file,
        additional_arguments={},
    )

    agent_2 = await train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        DEFAULT_STORIES_FILE,
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
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    from rasa.core.policies.ted_policy import TEDPolicy

    policies_config = {"policies": [{"name": TEDPolicy.__name__}]}

    policy_train = Mock()
    monkeypatch.setattr(TEDPolicy, "train", policy_train)

    interpreter = Mock(spec=RasaNLUInterpreter)

    await train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        DEFAULT_STORIES_FILE,
        str(tmp_path),
        interpreter=interpreter,
        policy_config=policies_config,
        additional_arguments={},
    )

    policy_train.assert_called_once()

    assert policy_train.call_count == 1
    _, _, kwargs = policy_train.mock_calls[0]
    assert kwargs["interpreter"] == interpreter
