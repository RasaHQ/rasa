from unittest.mock import patch

import numpy as np
import pytest

from rasa.shared.providers.embedding.embedding_client import EmbeddingClient
from rasa.shared.providers.embedding.huggingface_local_embedding_client import (
    HuggingFaceLocalEmbeddingClient,
)
import structlog

from tests.utilities import filter_logs


class TestHuggingFaceLocalEmbeddingClient:
    @pytest.fixture
    def client(self, tmpdir) -> HuggingFaceLocalEmbeddingClient:
        with patch(
            "rasa.shared.providers.embedding.huggingface_local_embedding_client"
            ".HuggingFaceLocalEmbeddingClient._init_client"
        ) as mock_init_client, patch(
            "rasa.shared.providers.embedding.huggingface_local_embedding_client"
            ".HuggingFaceLocalEmbeddingClient._embed_without_multiprocessing"
        ) as mock_embed_without_multiprocessing, patch(
            "rasa.shared.providers.embedding.huggingface_local_embedding_client"
            ".HuggingFaceLocalEmbeddingClient._embed_with_multiprocessing"
        ) as mock_embed_with_multiprocessing, patch(
            "rasa.shared.providers.embedding.huggingface_local_embedding_client"
            ".HuggingFaceLocalEmbeddingClient._validate_if_sentence_transformers_installed"
        ) as mock_validate_if_sentence_transformers_installed:
            mock_init_client.return_value = None
            mock_validate_if_sentence_transformers_installed.return_value = None
            embeddings = np.random.rand(384).reshape(1, 384)
            mock_embed_without_multiprocessing.return_value = embeddings
            mock_embed_with_multiprocessing.return_value = embeddings

            # Create and return the patched client
            client = HuggingFaceLocalEmbeddingClient(
                model="BAAI/bge-small-en-v1.5",
                cache_folder=str(tmpdir / "test_huggingface_local_embedding_client"),
            )
            yield client

    def test_conforms_to_protocol(
        self, client: HuggingFaceLocalEmbeddingClient
    ) -> None:
        assert isinstance(client, EmbeddingClient)

    def test_embed(
        self,
        client: HuggingFaceLocalEmbeddingClient,
    ) -> None:
        # Given
        test_doc = "this is a test doc."

        # When
        response = client.embed([test_doc])

        # Then
        assert len(response.data) == 1
        assert len(response.data[0]) == 384
        assert all(isinstance(item, float) for item in response.data[0])
        assert response.model == "BAAI/bge-small-en-v1.5"
        assert response.usage is None

    async def test_aembed(
        self,
        client: HuggingFaceLocalEmbeddingClient,
    ) -> None:
        # Given
        test_doc = "this is a test doc."

        # When
        response = await client.aembed([test_doc])

        # Then
        assert len(response.data) == 1
        assert len(response.data[0]) == 384
        assert all(isinstance(item, float) for item in response.data[0])
        assert response.model == "BAAI/bge-small-en-v1.5"
        assert response.usage is None

    def test_from_config_client_validation_error(self) -> None:
        # Given
        # Does not have the required `model` key
        config = {
            "provider": "huggingface",
        }

        # When
        with pytest.raises(ValueError), structlog.testing.capture_logs() as caplog:
            HuggingFaceLocalEmbeddingClient.from_config(config)

        expected_event = "validate_required_keys"
        expected_log_level = "error"
        logs = filter_logs(caplog, expected_event, expected_log_level, [])

        # Then
        assert len(logs) == 1

    def test_from_config_using_deprecated_type_settings(self):
        # Given
        config = {
            "type": "huggingface",
            "model": "BAAI/bge-small-en-v1.5",
        }

        # When
        with patch(
            "rasa.shared.providers.embedding.huggingface_local_embedding_client"
            ".HuggingFaceLocalEmbeddingClient._init_client"
        ) as mock_init_client, patch(
            "rasa.shared.providers.embedding.huggingface_local_embedding_client"
            ".HuggingFaceLocalEmbeddingClient._validate_if_sentence_transformers_installed"
        ) as mock_validate_if_sentence_transformers_installed:
            mock_init_client.return_value = None
            mock_validate_if_sentence_transformers_installed.return_value = None

            with pytest.warns(None) as record:
                client = HuggingFaceLocalEmbeddingClient.from_config(config)

        # Then
        future_warnings = [
            warning for warning in record if warning.category == FutureWarning
        ]
        assert len(future_warnings) == 2

        assert "type: huggingface" in str(future_warnings[0].message)
        assert "provider: huggingface_local" in str(future_warnings[0].message)

        assert "'type' is deprecated" in str(future_warnings[1].message)
        assert "'provider' instead" in str(future_warnings[1].message)

        assert client.config["model"] == "BAAI/bge-small-en-v1.5"
        assert client.config["provider"] == "huggingface_local"
