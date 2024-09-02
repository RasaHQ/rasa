from typing import Any, Callable, Dict, Type, Optional

from rasa.shared.constants import (
    AZURE_OPENAI_PROVIDER,
    HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER,
    OPENAI_PROVIDER,
)
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
from rasa.shared.providers._configs.azure_openai_client_config import (
    AzureOpenAIClientConfig,
)
from rasa.shared.providers._configs.default_litellm_client_config import (
    DefaultLiteLLMClientConfig,
)
from rasa.shared.providers._configs.huggingface_local_embedding_client_config import (
    HuggingFaceLocalEmbeddingClientConfig,
)
from rasa.shared.providers._configs.openai_client_config import OpenAIClientConfig


_provider_to_llm_client_mapping: Dict[str, Type[LLMClient]] = {
    OPENAI_PROVIDER: OpenAILLMClient,
    AZURE_OPENAI_PROVIDER: AzureOpenAILLMClient,
}

_provider_to_embedding_client_mapping: Dict[str, Type[EmbeddingClient]] = {
    OPENAI_PROVIDER: OpenAIEmbeddingClient,
    AZURE_OPENAI_PROVIDER: AzureOpenAIEmbeddingClient,
    HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER: HuggingFaceLocalEmbeddingClient,
}

# Define a type alias for the resolve aliases function signature
ResolveAliasesFn = Callable[[Dict[str, Any]], Dict[str, Any]]

_provider_to_client_config_resolve_aliases_fn_mapping: Dict[str, ResolveAliasesFn] = {
    OPENAI_PROVIDER: OpenAIClientConfig.resolve_config_aliases,
    AZURE_OPENAI_PROVIDER: AzureOpenAIClientConfig.resolve_config_aliases,
    HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER: HuggingFaceLocalEmbeddingClientConfig.resolve_config_aliases,  # noqa
}


def get_llm_client_from_provider(provider: Optional[str]) -> Type[LLMClient]:
    return _provider_to_llm_client_mapping.get(provider, DefaultLiteLLMClient)


def get_embedding_client_from_provider(provider: str) -> Type[EmbeddingClient]:
    return _provider_to_embedding_client_mapping.get(
        provider, DefaultLiteLLMEmbeddingClient
    )


def get_resolve_aliases_fn_from_provider(provider: str) -> Optional[ResolveAliasesFn]:
    return _provider_to_client_config_resolve_aliases_fn_mapping.get(
        provider, DefaultLiteLLMClientConfig.resolve_config_aliases
    )
