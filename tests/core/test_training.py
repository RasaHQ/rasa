import pytest

from rasa.core.interpreter import RegexInterpreter
from rasa.core.train import train
from rasa.core.agent import Agent
from rasa.core.policies.form_policy import FormPolicy

from rasa.core.training.dsl import StoryFileReader
from rasa.core.training.visualization import visualize_stories
from tests.core.conftest import DEFAULT_DOMAIN_PATH_WITH_SLOTS, DEFAULT_STORIES_FILE


async def test_story_visualization(default_domain, tmpdir):
    story_steps = await StoryFileReader.read_from_file(
        "data/test_stories/stories.md", default_domain, interpreter=RegexInterpreter()
    )
    out_file = tmpdir.join("graph.html").strpath
    generated_graph = await visualize_stories(
        story_steps,
        default_domain,
        output_file=out_file,
        max_history=3,
        should_merge_nodes=False,
    )

    assert len(generated_graph.nodes()) == 51

    assert len(generated_graph.edges()) == 56


async def test_story_visualization_with_merging(default_domain):
    story_steps = await StoryFileReader.read_from_file(
        "data/test_stories/stories.md", default_domain, interpreter=RegexInterpreter()
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


async def test_training_script(tmpdir):
    await train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        DEFAULT_STORIES_FILE,
        tmpdir.strpath,
        policy_config="data/test_config/max_hist_config.yml",
        interpreter=RegexInterpreter(),
        additional_arguments={},
    )
    assert True


async def test_training_script_without_max_history_set(tmpdir):
    await train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        DEFAULT_STORIES_FILE,
        tmpdir.strpath,
        interpreter=RegexInterpreter(),
        policy_config="data/test_config/no_max_hist_config.yml",
        additional_arguments={},
    )

    agent = Agent.load(tmpdir.strpath)
    for policy in agent.policy_ensemble.policies:
        if hasattr(policy.featurizer, "max_history"):
            if type(policy) == FormPolicy:
                assert policy.featurizer.max_history == 2
            else:
                assert (
                    policy.featurizer.max_history
                    == policy.featurizer.MAX_HISTORY_DEFAULT
                )


async def test_training_script_with_max_history_set(tmpdir):
    await train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        DEFAULT_STORIES_FILE,
        tmpdir.strpath,
        interpreter=RegexInterpreter(),
        policy_config="data/test_config/max_hist_config.yml",
        additional_arguments={},
    )
    agent = Agent.load(tmpdir.strpath)
    for policy in agent.policy_ensemble.policies:
        if hasattr(policy.featurizer, "max_history"):
            if type(policy) == FormPolicy:
                assert policy.featurizer.max_history == 2
            else:
                assert policy.featurizer.max_history == 5


async def test_training_script_with_restart_stories(tmpdir):
    await train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        "data/test_stories/stories_restart.md",
        tmpdir.strpath,
        interpreter=RegexInterpreter(),
        policy_config="data/test_config/max_hist_config.yml",
        additional_arguments={},
    )
    assert True


def configs_for_random_seed_test():
    # define the configs for the random_seed tests
    return [
        "data/test_config/keras_random_seed.yaml",
        "data/test_config/embedding_random_seed.yaml",
    ]


@pytest.mark.parametrize("config_file", configs_for_random_seed_test())
async def test_random_seed(tmpdir, config_file):
    # set random seed in config file to
    # generate a reproducible training result

    agent_1 = await train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        DEFAULT_STORIES_FILE,
        tmpdir.strpath + "1",
        interpreter=RegexInterpreter(),
        policy_config=config_file,
        additional_arguments={},
    )

    agent_2 = await train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        DEFAULT_STORIES_FILE,
        tmpdir.strpath + "2",
        interpreter=RegexInterpreter(),
        policy_config=config_file,
        additional_arguments={},
    )

    processor_1 = agent_1.create_processor()
    processor_2 = agent_2.create_processor()

    probs_1 = await processor_1.predict_next("1")
    probs_2 = await processor_2.predict_next("2")
    assert probs_1["confidence"] == probs_2["confidence"]
