import pytest
from pytest import MonkeyPatch

from rasa.shared.constants import OPENAI_API_BASE_ENV_VAR, OPENAI_API_KEY_ENV_VAR
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.providers.llm.openai_llm_client import OpenAILLMClient, OPENAI_PROVIDER


class TestOpenAILLMClient:
    @pytest.fixture
    def client(self, monkeypatch: MonkeyPatch) -> LLMClient:
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        return OpenAILLMClient(
            model="test_model",
        )

    def test_conforms_to_protocol(self, client: LLMClient, monkeypatch: MonkeyPatch):
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        assert isinstance(client, LLMClient)

    def test_init_fetches_from_environment_variables(self, monkeypatch: MonkeyPatch):
        # Given

        # Set the environment variables
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        monkeypatch.setenv(OPENAI_API_BASE_ENV_VAR, "https://my.api.base.com/my_model")

        # When
        client = OpenAILLMClient(model="test_model", api_base=None)

        # Then
        assert client.api_base == "https://my.api.base.com/my_model"

    @pytest.mark.parametrize(
        "config, expected_model, expected_api_base, expected_model_parameters",
        [
            (
                {"model": "test_model", "temperature": 0.2, "max_tokens": 1000},
                "test_model",
                None,
                {"temperature": 0.2, "max_tokens": 1000},
            ),
            # use deprecated alias for model
            ({"model_name": "test_model"}, "test_model", None, {}),
            # use api base
            (
                {"model": "test_model", "api_base": "https://my.api.base.com/my_model"},
                "test_model",
                "https://my.api.base.com/my_model",
                {},
            ),
            # deprecated alias to api base
            (
                {
                    "model": "test_model",
                    "openai_api_base": "https://my.api.base.com/my_model",
                },
                "test_model",
                "https://my.api.base.com/my_model",
                {},
            ),
        ],
    )
    def test_from_config(
        self,
        config: dict,
        expected_model: str,
        expected_api_base: str,
        expected_model_parameters: dict,
        monkeypatch: MonkeyPatch,
    ):
        # Given
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")

        # When
        client = OpenAILLMClient.from_config(config)

        # Then
        assert client.model == expected_model
        assert client.api_base == expected_api_base
        assert client.provider == OPENAI_PROVIDER
        assert len(client.model_parameters) == len(expected_model_parameters)
        for parameter_key, parameter_value in expected_model_parameters.items():
            assert parameter_key in client.model_parameters
            assert client.model_parameters[parameter_key] == parameter_value

    def test_completion(
        self,
        client: OpenAILLMClient,
        monkeypatch: MonkeyPatch,
    ) -> None:
        # Given
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        test_prompt = "Hello, this is a test prompt."
        test_response = "Hello, this is mocked response!"
        # LiteLLM supports mocking response for testing purposes
        client._model_parameters = {"mock_response": test_response}

        # When
        response = client.completion([test_prompt])

        # Then
        assert response.choices == [test_response]
        assert response.model == client.model
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    async def test_acompletion(
        self, client: OpenAILLMClient, monkeypatch: MonkeyPatch
    ) -> None:
        # Given
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")

        test_prompt = "Hello, this is a test prompt."
        test_response = "Hello, this is mocked response!"
        # LiteLLM supports mocking response for testing purposes
        client._model_parameters = {"mock_response": test_response}

        # When
        response = await client.acompletion([test_prompt])

        # Then
        assert response.choices == [test_response]
        assert response.model == client.model
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0
