from typing import Dict, Type, Optional

from rasa.shared.providers.embedding.azure_openai_embedding_client import (
    AzureOpenAIEmbeddingClient,
)
from rasa.shared.providers.embedding.default_litellm_embedding_client import (
    DefaultLiteLLMEmbeddingClient,
)
from rasa.shared.providers.embedding.embedding_client import EmbeddingClient
from rasa.shared.providers.embedding.huggingface_local_embedding_client import (
    HuggingFaceLocalEmbeddingClient,
)
from rasa.shared.providers.embedding.openai_embedding_client import (
    OpenAIEmbeddingClient,
)
from rasa.shared.providers.llm.azure_openai_llm_client import AzureOpenAILLMClient
from rasa.shared.providers.llm.default_litellm_llm_client import DefaultLiteLLMClient
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.providers.llm.openai_llm_client import OpenAILLMClient

OPENAI_PROVIDER = "openai"
AZURE_OPENAI_PROVIDER = "azure"
HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER = "huggingface_local"

_provider_to_llm_client_mapping: Dict[str, Type[LLMClient]] = {
    OPENAI_PROVIDER: OpenAILLMClient,
    AZURE_OPENAI_PROVIDER: AzureOpenAILLMClient,
}

_provider_to_embedding_client_mapping: Dict[str, Type[EmbeddingClient]] = {
    OPENAI_PROVIDER: OpenAIEmbeddingClient,
    AZURE_OPENAI_PROVIDER: AzureOpenAIEmbeddingClient,
    HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER: HuggingFaceLocalEmbeddingClient,
}


def get_llm_client_from_provider(provider: Optional[str]) -> Type[LLMClient]:
    return _provider_to_llm_client_mapping.get(provider, DefaultLiteLLMClient)


def get_embedding_client_from_provider(provider: str) -> Type[EmbeddingClient]:
    return _provider_to_embedding_client_mapping.get(
        provider, DefaultLiteLLMEmbeddingClient
    )
