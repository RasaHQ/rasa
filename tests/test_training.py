from rasa_core.interpreter import RegexInterpreter
from rasa_core.train import train_dialogue_model
from rasa_core.agent import Agent
from rasa_core.policies.form_policy import FormPolicy

from rasa_core.training.dsl import StoryFileReader
from rasa_core.training.visualization import visualize_stories
from tests.conftest import DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE


def test_story_visualization_script():
    from rasa_core.visualize import create_argument_parser
    assert create_argument_parser() is not None


def test_story_visualization(default_domain, tmpdir):
    story_steps = StoryFileReader.read_from_file(
        "data/test_stories/stories.md", default_domain,
        interpreter=RegexInterpreter())
    out_file = tmpdir.join("graph.html").strpath
    generated_graph = visualize_stories(story_steps, default_domain,
                                        output_file=out_file,
                                        max_history=3,
                                        should_merge_nodes=False)

    assert len(generated_graph.nodes()) == 51

    assert len(generated_graph.edges()) == 56


def test_story_visualization_with_merging(default_domain):
    story_steps = StoryFileReader.read_from_file(
        "data/test_stories/stories.md", default_domain,
        interpreter=RegexInterpreter())
    generated_graph = visualize_stories(story_steps, default_domain,
                                        output_file=None,
                                        max_history=3,
                                        should_merge_nodes=True)
    assert 15 < len(generated_graph.nodes()) < 33

    assert 20 < len(generated_graph.edges()) < 33


def test_training_script(tmpdir):
    train_dialogue_model(DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE,
                         tmpdir.strpath,
                         policy_config='data/test_config/max_hist_config.yml',
                         interpreter=RegexInterpreter(),
                         kwargs={})
    assert True


def test_training_script_without_max_history_set(tmpdir):
    train_dialogue_model(
        DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE,
        tmpdir.strpath,
        interpreter=RegexInterpreter(),
        policy_config='data/test_config/no_max_hist_config.yml',
        kwargs={})
    agent = Agent.load(tmpdir.strpath)
    for policy in agent.policy_ensemble.policies:
        if hasattr(policy.featurizer, 'max_history'):
            if type(policy) == FormPolicy:
                assert policy.featurizer.max_history == 2
            else:
                assert (policy.featurizer.max_history ==
                        policy.featurizer.MAX_HISTORY_DEFAULT)


def test_training_script_with_max_history_set(tmpdir):
    train_dialogue_model(DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE,
                         tmpdir.strpath,
                         interpreter=RegexInterpreter(),
                         policy_config='data/test_config/max_hist_config.yml',
                         kwargs={})
    agent = Agent.load(tmpdir.strpath)
    for policy in agent.policy_ensemble.policies:
        if hasattr(policy.featurizer, 'max_history'):
            if type(policy) == FormPolicy:
                assert policy.featurizer.max_history == 2
            else:
                assert policy.featurizer.max_history == 5


def test_training_script_with_restart_stories(tmpdir):
    train_dialogue_model(DEFAULT_DOMAIN_PATH,
                         "data/test_stories/stories_restart.md",
                         tmpdir.strpath,
                         interpreter=RegexInterpreter(),
                         policy_config='data/test_config/max_hist_config.yml',
                         kwargs={})
    assert True
