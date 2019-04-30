import pytest

from rasa import server
from rasa.core.agent import Agent
from rasa.core.channels import RestInput, channel
from rasa.core.policies import AugmentedMemoizationPolicy
from rasa.train import train_async
from tests.core.conftest import DEFAULT_STORIES_FILE


@pytest.fixture
async def default_agent(tmpdir_factory) -> Agent:
    model_path = tmpdir_factory.mktemp("model").strpath

    agent = Agent(
        "data/test_domains/default.yml",
        policies=[AugmentedMemoizationPolicy(max_history=3)],
    )

    training_data = await agent.load_data(DEFAULT_STORIES_FILE)
    agent.train(training_data)
    agent.persist(model_path)
    return agent


@pytest.fixture()
async def trained_rasa_model(
    default_domain_path, default_config, default_nlu_data, default_stories_file
):
    trained_stack_model_path = await train_async(
        domain=default_domain_path,
        config=default_config,
        training_files=[default_nlu_data, default_stories_file],
    )

    return trained_stack_model_path


@pytest.fixture
async def rasa_server(default_agent):
    app = server.create_app(agent=default_agent)
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
async def rasa_server_secured(default_agent):
    app = server.create_app(agent=default_agent, auth_token="rasa", jwt_secret="core")
    channel.register([RestInput()], app, "/webhooks/")
    return app
