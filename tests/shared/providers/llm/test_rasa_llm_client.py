import pytest
from unittest.mock import Mock, AsyncMock
from pytest import MonkeyPatch

from rasa.shared.constants import OPENAI_PROVIDER, RASA_PROVIDER
from rasa.shared.providers.llm.rasa_llm_client import RasaLLMClient
from rasa.shared.providers._configs.rasa_llm_client_config import RasaLLMClientConfig
from rasa.shared.providers.llm.llm_response import LLMResponse

class TestRasaLLMClient:

    @pytest.fixture
    def client(self) -> RasaLLMClient:
        return RasaLLMClient(
            model="rasa/cmd_gen_codellama_13b_calm_demo",
            api_base="https://huggingface-proxy.rasa-e2e.workers.dev",
        )

    @pytest.fixture
    def mock_retrieve_license(self, monkeypatch: MonkeyPatch) -> Mock:
        mock = Mock(return_value='mock-license')
        monkeypatch.setattr('rasa.shared.providers.llm.rasa_llm_client.retrieve_license_from_env', mock)
        return mock

    def test_from_config_valid(self) -> None:
        config = {
            'provider': RASA_PROVIDER,            
            'model': 'rasa/cmd_gen_codellama_13b_calm_demo',
            'api_base': 'https://huggingface-proxy.rasa-e2e.workers.dev',
        }
        client = RasaLLMClient.from_config(config)
        assert isinstance(client, RasaLLMClient)
        assert client.model == 'rasa/cmd_gen_codellama_13b_calm_demo'
        assert client.api_base == 'https://huggingface-proxy.rasa-e2e.workers.dev'
        assert client.provider == RASA_PROVIDER

    def test_from_config_invalid(self) -> None:
        invalid_config = {}
        with pytest.raises(ValueError):
            RasaLLMClient.from_config(invalid_config)

    def test_completion(self, client: RasaLLMClient, monkeypatch: MonkeyPatch) -> None:
        mock_completion = Mock(return_value=LLMResponse(id="test_id", created=1234567890, choices=["Test response"]))
        monkeypatch.setattr(RasaLLMClient, 'completion', mock_completion)
        response = client.completion(["Test prompt"])
        assert isinstance(response, LLMResponse)
        assert response.choices == ["Test response"]

    @pytest.mark.asyncio
    async def test_acompletion(self, client: RasaLLMClient, monkeypatch: MonkeyPatch) -> None:
        mock_acompletion = AsyncMock(return_value=LLMResponse(id="test_id", created=1234567890, choices=["Test response"]))
        monkeypatch.setattr(RasaLLMClient, 'acompletion', mock_acompletion)
        response = await client.acompletion(["Test prompt"])
        assert isinstance(response, LLMResponse)
        assert response.choices == ["Test response"]

    def test_litellm_model_name(self, client: RasaLLMClient) -> None:
        assert client._litellm_model_name == f"{OPENAI_PROVIDER}/rasa/cmd_gen_codellama_13b_calm_demo"

    def test_litellm_extra_parameters(self, client: RasaLLMClient) -> None:
        assert client._litellm_extra_parameters == {}

    def test_completion_fn_args(self, client: RasaLLMClient, mock_retrieve_license: Mock) -> None:
        fn_args = client._completion_fn_args
        assert fn_args["model"] == f"{OPENAI_PROVIDER}/rasa/cmd_gen_codellama_13b_calm_demo"
        assert fn_args["api_base"] == "https://huggingface-proxy.rasa-e2e.workers.dev"
        assert fn_args["api_key"] == "mock-license"

    def test_config(self, client: RasaLLMClient) -> None:
        expected_config = {
            "model": "rasa/cmd_gen_codellama_13b_calm_demo",
            "api_base": "https://huggingface-proxy.rasa-e2e.workers.dev",
            "provider": RASA_PROVIDER,
        }
        assert client.config == expected_config

    def test_config_roundtrip(self):
        # Initial configuration
        initial_config = {
            'provider': RASA_PROVIDER,
            'model': 'rasa/cmd_gen_codellama_13b_calm_demo',
            'api_base': 'https://huggingface-proxy.rasa-e2e.workers.dev',
            'extra_param1': 'value1',
            'extra_param2': 'value2',
        }

        # Create RasaLLMClientConfig from initial config
        config1 = RasaLLMClientConfig.from_dict(initial_config)

        # Convert config1 to dict
        dict1 = config1.to_dict()

        # Create new RasaLLMClientConfig from dict1
        config2 = RasaLLMClientConfig.from_dict(dict1)

        # Convert config2 to dict
        dict2 = config2.to_dict()

        # Assert that all initial config keys are present in both dict1 and dict2
        for key, value in initial_config.items():
            assert key in dict1, f"Key '{key}' missing in dict1"
            assert key in dict2, f"Key '{key}' missing in dict2"
            assert dict1[key] == value, f"Value mismatch for key '{key}' in dict1"
            assert dict2[key] == value, f"Value mismatch for key '{key}' in dict2"

        # Assert that no extra keys were added
        assert set(dict1.keys()) == set(initial_config.keys()), "Extra keys in dict1"
        assert set(dict2.keys()) == set(initial_config.keys()), "Extra keys in dict2"

        # Compare the two dicts directly
        assert dict1 == dict2, "dict1 and dict2 are not identical"

        # Create a RasaLLMClient from dict2
        client = RasaLLMClient.from_config(dict2)

        # Get config from client
        client_config = client.config

        # Assert that all initial config keys are present in client_config
        for key, value in initial_config.items():
            assert key in client_config, f"Key '{key}' missing in client_config"
            assert client_config[key] == value, f"Value mismatch for key '{key}' in client_config"

        # Assert that no extra keys were added
        assert set(client_config.keys()) == set(initial_config.keys()), "Extra keys in client_config"

        # Test roundtrip through RasaLLMClient
        final_dict = RasaLLMClientConfig.from_dict(client_config).to_dict()
        assert final_dict == initial_config, "Final dict does not match initial config after full roundtrip"