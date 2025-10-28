import json
import os
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Union
from unittest.mock import MagicMock, Mock, patch

import boto3
import pytest
from moto import mock_aws
from pytest import CaptureFixture, MonkeyPatch

from rasa.core.agent import Agent, load_agent
from rasa.core.config.configuration import Configuration
from rasa.core.persistor import AWSPersistor, RemoteStorageType
from rasa.e2e_test.e2e_test_case import Fixture, TestCase, TestStep
from rasa.e2e_test.e2e_test_runner import E2ETestRunner
from tests.conftest import TrainedAsync

# Path for subagents used in tests
SUB_AGENTS_PATH = "sub_agents"


@pytest.fixture
def mock_model(tmp_path: Path) -> Path:
    """Name of the model to use for testing."""
    model_file = tmp_path / "my-model.tar.gz"

    tarred_file = tmp_path / "dummy_file"
    tarred_file.touch()

    with tarfile.open(model_file, "w:gz") as tar:
        tar.add(tarred_file, arcname=tarred_file.name)

    return model_file


@pytest.fixture
def bucket_name() -> Text:
    """Name of the bucket to use for testing."""
    return "rasa-test"


@pytest.fixture
def region_name() -> Text:
    """Name of the region to use for testing."""
    return "us-east-1"


@mock_aws
def create_user_with_access_key_and_attached_policy(region_name: Text) -> Any:
    """Create a user and an access key for them."""
    client = boto3.client("iam", region_name=region_name)
    # deepcode ignore NoHardcodedCredentials/test: Test secret
    client.create_user(UserName="test_user")

    policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllObjectActions",
                "Effect": "Allow",
                "Action": "s3:*Object",
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
            }
        ],
    }

    policy_arn = client.create_policy(
        PolicyName="test_policy", PolicyDocument=json.dumps(policy_document)
    )["Policy"]["Arn"]
    client.attach_user_policy(UserName="test_user", PolicyArn=policy_arn)

    return client.create_access_key(UserName="test_user")["AccessKey"]


@pytest.fixture
def aws_environment_variables(
    bucket_name: Text,
    region_name: Text,
) -> None:
    """Set AWS environment variables for testing."""
    os.environ["BUCKET_NAME"] = bucket_name
    os.environ["AWS_DEFAULT_REGION"] = region_name

    access_key = create_user_with_access_key_and_attached_policy(region_name)

    os.environ["AWS_ACCESS_KEY_ID"] = access_key["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = access_key["SecretAccessKey"]
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"

    os.environ["TEST_SERVER_MODE"] = "true"


def mock_load_model(
    self: Any, model_path: Union[Text, Path], fingerprint: Optional[Text] = None
) -> None:
    """Mock load model function."""

    class MockProcessor:
        def __init__(self, model_path: Any) -> None:
            self.model_path = Path(model_path)

    self.processor = MockProcessor(model_path)


@mock_aws
def test_e2e_test_runner_load_agent_from_remote_storage(
    mock_model: Path,
    bucket_name: Text,
    region_name: Text,
    aws_environment_variables: None,
    monkeypatch: MonkeyPatch,
) -> None:
    model_name = mock_model.name

    conn = boto3.resource("s3", region_name=region_name)
    # We need to create the bucket in Moto's 'virtual' AWS account
    # prior to AWSPersistor instantiation
    conn.create_bucket(Bucket=bucket_name)
    # upload model file to bucket
    with open(str(mock_model), "rb") as f:
        conn.meta.client.upload_fileobj(f, bucket_name, model_name)

    def mock_aws_persistor(name: Text) -> AWSPersistor:
        aws_persistor = AWSPersistor(bucket_name, region_name=region_name)
        monkeypatch.setattr(aws_persistor, "s3", conn)
        monkeypatch.setattr(aws_persistor, "bucket", conn.Bucket(bucket_name))
        return aws_persistor

    monkeypatch.setattr("rasa.core.persistor.get_persistor", mock_aws_persistor)
    monkeypatch.setattr("rasa.core.agent.Agent.load_model", mock_load_model)

    # Mock Configuration.get_instance() to avoid initialization requirement
    mock_config = MagicMock()
    mock_config.available_agents = None
    monkeypatch.setattr(
        "rasa.core.config.configuration.Configuration.get_instance",
        lambda: mock_config,
    )

    test_runner = E2ETestRunner(
        model_path=model_name,
        remote_storage=RemoteStorageType.AWS,
        sub_agents_path=SUB_AGENTS_PATH,
    )

    assert isinstance(test_runner.agent, Agent)
    assert test_runner.agent.remote_storage == RemoteStorageType.AWS

    assert test_runner.agent.processor is not None
    assert test_runner.agent.model_name is not None


@pytest.fixture(scope="session")
@patch("langchain_community.vectorstores.faiss.FAISS.from_documents")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_llm_client")
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_embedder")
async def trained_custom_action_session_start_calm_bot(
    mock_try_instantiate_llm_client: Mock,
    mock_try_instantiate_embedder: Mock,
    mock_flow_search_create_embedder: Mock,
    mock_from_documents: Mock,
    trained_async: TrainedAsync,
) -> Text:
    parent_folder = "data/test_e2e_test_runner_with_customised_action_session_start"
    domain_path = f"{parent_folder}/domain.yml"
    config_path = f"{parent_folder}/config.yml"
    data_path = f"{parent_folder}/data"

    mock_try_instantiate_llm_client.return_value = Mock()
    mock_flow_search_create_embedder.return_value = Mock()
    mock_from_documents.return_value = Mock()
    mock_try_instantiate_embedder.return_value = Mock()
    return await trained_async(
        domain=domain_path,
        config=config_path,
        training_files=[
            data_path,
        ],
    )


@pytest.mark.parametrize(
    "dispatched_response, fixture_names",
    [
        ({"response": "utter_greet"}, ["test_fixture"]),
        ({"text": "Hello World!"}, ["test_fixture"]),
        ({"response": "utter_greet"}, []),
        ({"text": "Hello World!"}, []),
    ],
)
@patch("langchain_community.vectorstores.faiss.FAISS.load_local")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
async def test_e2e_test_runner_with_customized_action_session_start(
    mock_flow_search_create_embedder: Mock,
    mock_load_local: Mock,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture,
    dispatched_response: Dict[str, str],
    fixture_names: List[str],
    trained_custom_action_session_start_calm_bot: str,
) -> None:
    mock_flow_search_create_embedder.return_value = Mock()
    mock_load_local.return_value = Mock()

    endpoints_path = (
        "data/test_e2e_test_runner_with_customised_action_session_start/endpoints.yml"
    )
    endpoints = Configuration.initialise_endpoints(
        endpoints_path=Path(endpoints_path)
    ).endpoints
    test_agent = await load_agent(
        model_path=trained_custom_action_session_start_calm_bot, endpoints=endpoints
    )

    def mock_init(self, *args, **kwargs) -> None:
        self.agent = test_agent
        self.llm_judge_config = MagicMock()

    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.__init__", mock_init
    )

    async def mock_run(self, *args, **kwargs) -> Dict[str, Any]:
        return {"responses": [dispatched_response]}

    # Mock RetryCustomActionExecutor instead, HTTPCustomActionExecutor is now wrapped
    monkeypatch.setattr(
        "rasa.core.actions.custom_action_executor.RetryCustomActionExecutor.run",
        mock_run,
    )

    test_runner = E2ETestRunner()
    result = await test_runner.run_tests(
        input_test_cases=[
            TestCase(
                steps=[
                    TestStep.from_dict({"user": "Hi!"}),
                ],
                name="test_e2e_test_runner_with_customized_action_session_start",
                file="data/test_e2e_test_runner_with_customised_action_session_start/e2e_test.yml",
                fixture_names=fixture_names,
            )
        ],
        input_fixtures=[
            Fixture.from_dict({"test_fixture": [{"add_contact_handle": "test"}]})
        ],
        input_metadata=[],
    )

    assert result[0].pass_status
    captured = capsys.readouterr()
    assert (
        "Encountered an exception while running action 'action_session_start'"
        not in captured.out
    )
