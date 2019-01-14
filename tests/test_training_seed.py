from rasa_core.interpreter import RegexInterpreter
from rasa_core.train import train_dialogue_model
from rasa_core.agent import Agent
from rasa_core.policies.form_policy import FormPolicy

from rasa_core.training.dsl import StoryFileReader
from rasa_core.training.visualization import visualize_stories
from tests.conftest import DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE


def test_training_script_with_random_seed(tmpdir):
    from rasa_core import training
    for i in range(100):
        # set kwargs reproducible and set random seed in config file , which will
        # generate reproducible training result.
        agent_1 = train_dialogue_model(
            DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE,
            tmpdir.strpath + "1",
            interpreter=RegexInterpreter(),
            policy_config='data/test_config/random_seed.yaml',
            kwargs={})

        agent_2 = train_dialogue_model(
            DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE,
            tmpdir.strpath + "2",
            interpreter=RegexInterpreter(),
            policy_config='data/test_config/random_seed.yaml',
            kwargs={})
            
        processor_1 = agent_1.create_processor()
        processor_2 = agent_2.create_processor()

        probs_1 = processor_1.predict_next("1")
        probs_2 = processor_2.predict_next("2")
        assert probs_1["confidence"] == probs_2["confidence"]


