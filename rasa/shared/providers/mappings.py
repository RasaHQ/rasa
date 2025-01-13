from typing import Dict, Optional, Type

from rasa.shared.constants import (
    AZURE_OPENAI_PROVIDER,
    HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER,
    OPENAI_PROVIDER,
    RASA_PROVIDER,
    SELF_HOSTED_PROVIDER,
    SELF_HOSTED_VLLM_PREFIX,
)
from rasa.shared.providers._configs.azure_openai_client_config import (
    AzureOpenAIClientConfig,
)
from rasa.shared.providers._configs.client_config import ClientConfig
from rasa.shared.providers._configs.default_litellm_client_config import (
    DefaultLiteLLMClientConfig,
)
from rasa.shared.providers._configs.huggingface_local_embedding_client_config import (
    HuggingFaceLocalEmbeddingClientConfig,
)
from rasa.shared.providers._configs.openai_client_config import OpenAIClientConfig
from rasa.shared.providers._configs.rasa_llm_client_config import RasaLLMClientConfig
from rasa.shared.providers._configs.self_hosted_llm_client_config import (
    SelfHostedLLMClientConfig,
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
from rasa.shared.providers.llm.rasa_llm_client import RasaLLMClient
from rasa.shared.providers.llm.self_hosted_llm_client import SelfHostedLLMClient

_provider_to_llm_client_mapping: Dict[str, Type[LLMClient]] = {
    OPENAI_PROVIDER: OpenAILLMClient,
    AZURE_OPENAI_PROVIDER: AzureOpenAILLMClient,
    SELF_HOSTED_PROVIDER: SelfHostedLLMClient,
    RASA_PROVIDER: RasaLLMClient,
}

_provider_to_embedding_client_mapping: Dict[str, Type[EmbeddingClient]] = {
    OPENAI_PROVIDER: OpenAIEmbeddingClient,
    AZURE_OPENAI_PROVIDER: AzureOpenAIEmbeddingClient,
    HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER: HuggingFaceLocalEmbeddingClient,
}

_provider_to_client_config_class_mapping: Dict[str, Type] = {
    OPENAI_PROVIDER: OpenAIClientConfig,
    AZURE_OPENAI_PROVIDER: AzureOpenAIClientConfig,
    HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER: HuggingFaceLocalEmbeddingClientConfig,
    SELF_HOSTED_PROVIDER: SelfHostedLLMClientConfig,
    RASA_PROVIDER: RasaLLMClientConfig,
}


_provider_to_prefix_mapping: Dict[str, str] = {
    # Specify the provider name as the key and its corresponding prefix as the value
    # for providers where the prefix differs from the provider name.
    SELF_HOSTED_PROVIDER: SELF_HOSTED_VLLM_PREFIX,
    RASA_PROVIDER: OPENAI_PROVIDER,
}


def get_llm_client_from_provider(provider: Optional[str]) -> Type[LLMClient]:
    return _provider_to_llm_client_mapping.get(provider, DefaultLiteLLMClient)


def get_embedding_client_from_provider(provider: str) -> Type[EmbeddingClient]:
    return _provider_to_embedding_client_mapping.get(
        provider, DefaultLiteLLMEmbeddingClient
    )


def get_client_config_class_from_provider(provider: str) -> Type[ClientConfig]:
    return _provider_to_client_config_class_mapping.get(
        provider, DefaultLiteLLMClientConfig
    )


def get_prefix_from_provider(provider: str) -> str:
    return _provider_to_prefix_mapping.get(provider, provider)
