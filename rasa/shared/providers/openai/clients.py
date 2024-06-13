import os
import ssl
from typing import List, Optional, Sequence, Any

import certifi
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.openai import OpenAIChat

from rasa.shared.constants import (
    REQUESTS_CA_BUNDLE_ENV_VAR,
    REQUESTS_SSL_CONTEXT_PURPOSE_ENV_VAR,
)
from rasa.shared.providers.openai.session_handler import OpenAISessionHandler

CERTIFICATE_FILE = os.environ.get(REQUESTS_CA_BUNDLE_ENV_VAR, certifi.where())
SSL_PURPOSE = os.environ.get(
    REQUESTS_SSL_CONTEXT_PURPOSE_ENV_VAR, ssl.Purpose.SERVER_AUTH
)


class AioHTTPSessionAzureChatOpenAI(AzureChatOpenAI):
    async def apredict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        async with OpenAISessionHandler(CERTIFICATE_FILE, SSL_PURPOSE):
            return await super().apredict(text, stop=stop, **kwargs)


class AioHTTPSessionOpenAIChat(OpenAIChat):
    async def apredict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        async with OpenAISessionHandler(CERTIFICATE_FILE, SSL_PURPOSE):
            return await super().apredict(text, stop=stop, **kwargs)


class AioHTTPSessionOpenAIEmbeddings(OpenAIEmbeddings):
    async def aembed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        async with OpenAISessionHandler(CERTIFICATE_FILE, SSL_PURPOSE):
            return await super().aembed_documents(texts, chunk_size)
