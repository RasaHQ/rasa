import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

import openai
import structlog
from openai.types.chat import ChatCompletion

from rasa.builder import config
from rasa.builder.copilot.constants import ROLE_USER
from rasa.builder.document_retrieval.constants import (
    INKEEP_API_KEY_ENV_VAR,
    INKEEP_DOCUMENT_RETRIEVAL_MODEL,
    INKEEP_RAG_RESPONSE_SCHEMA_PATH,
)
from rasa.builder.document_retrieval.models import Document
from rasa.builder.exceptions import DocumentRetrievalError
from rasa.builder.telemetry.copilot_langfuse_telemetry import CopilotLangfuseTelemetry
from rasa.shared.utils.io import read_json_file

structlogger = structlog.get_logger()


class InKeepDocumentRetrieval:
    """Handles the document retrieval from InKeep AI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        self._rag_schema = read_json_file(INKEEP_RAG_RESPONSE_SCHEMA_PATH)
        self._api_key = api_key or os.getenv(INKEEP_API_KEY_ENV_VAR)

    @property
    def api_key(self) -> str:
        """Resolve the correct API key/token based on proxy usage."""
        using_proxy = bool(config.HELLO_LLM_PROXY_BASE_URL)
        if using_proxy:
            if not config.RASA_PRO_LICENSE:
                raise DocumentRetrievalError(
                    "HELLO_LLM_PROXY_BASE_URL is set but RASA_PRO_LICENSE is missing. "
                    "Proxy requires a Rasa Pro license token for authentication."
                )
            return config.RASA_PRO_LICENSE

        if not self._api_key:
            raise DocumentRetrievalError(
                "INKEEP_API_KEY is missing. Provide it via env INKEEP_API_KEY "
                "or pass api_key= to InKeepDocumentRetrieval."
            )
        return self._api_key

    async def retrieve_documents(
        self, query: str, temperature: float = 0.0, timeout: float = 30.0
    ) -> List[Document]:
        """Retrieve relevant documents using InKeep AI based on the given query.

        Args:
            query: The search query
            temperature: Controls randomness in generation (0.0 for deterministic)
            timeout: Timeout for the API call

        Returns:
            List of Document objects containing retrieved content

        Raises:
            DocumentRetrievalError: When document retrieval fails due to:
                - Empty response from InKeep AI API
                - OpenAI API errors (authentication, rate limiting, etc.)
                - Request timeout
                - Unexpected errors during API communication
        """
        try:
            response = await self._call_inkeep_rag_api(
                query=query,
                temperature=temperature,
                timeout=timeout,
            )
            documents = self._parse_documents_from_response(response)
            return documents
        except DocumentRetrievalError as e:
            structlogger.error(
                "inkeep_document_retrieval.retrieve_documents.error",
                event_info="InKeep Document Retrieval: Error",
                query=query,
                error=str(e),
            )
            raise e

    @CopilotLangfuseTelemetry.trace_document_retrieval_generation
    async def _call_inkeep_rag_api(
        self, query: str, temperature: float, timeout: float
    ) -> ChatCompletion:
        """Call InKeep AI RAG's API endpoint and return the response content.

        Args:
            query: The search query to send to InKeep
            temperature: Controls randomness in generation (0.0 for deterministic)
            timeout: Timeout for the API call

        Returns:
            The response content from InKeep AI. The response is made of the retrieved
            documents.

        Raises:
            LLMGenerationError: If the API call fails or returns invalid response
        """
        try:
            async with self._get_client() as client:
                response = await client.chat.completions.create(
                    model=INKEEP_DOCUMENT_RETRIEVAL_MODEL,
                    messages=[{"role": ROLE_USER, "content": query}],
                    temperature=temperature,
                    timeout=timeout,
                    response_format={
                        "type": "json_schema",
                        "json_schema": self._rag_schema,
                    },
                )

                if not response.choices[0].message.content:
                    structlogger.warning(
                        "inkeep_document_retrieval.empty_response",
                        event_info="InKeep AI returned an empty response. ",
                        query=query,
                        response_content=response.choices[0].message.content,
                    )
                    raise DocumentRetrievalError(
                        "InKeep Document Retrieval: Empty response"
                    )

                return response

        except openai.OpenAIError as e:
            structlogger.error(
                "inkeep_document_retrieval.call_inkeep_rag_api.api_error",
                event_info="InKeep Document Retrieval: API error",
                query=query,
                error=e,
            )
            raise DocumentRetrievalError(f"InKeep Document Retrieval: API error: {e}")
        except asyncio.TimeoutError as e:
            structlogger.error(
                "inkeep_document_retrieval.call_inkeep_rag_api.timeout_error",
                event_info="InKeep Document Retrieval: Timeout error",
                query=query,
                error=e,
            )
            raise DocumentRetrievalError(f"InKeep AI request timed out: {e}")
        except Exception as e:
            structlogger.error(
                "inkeep_document_retrieval.call_inkeep_rag_api.error",
                event_info="InKeep Document Retrieval: Error",
                query=query,
                error=e,
            )
            raise DocumentRetrievalError(
                f"InKeep Document Retrieval: Unexpected error: {e}"
            )

    @asynccontextmanager
    async def _get_client(self) -> AsyncGenerator[openai.AsyncOpenAI, None]:
        """Get or create client that handles the API calls to InKeep AI."""
        # Ensure trailing slash to match client expectations/tests
        base_url = f"{config.INKEEP_BASE_URL.rstrip('/')}/"
        client = openai.AsyncOpenAI(base_url=base_url, api_key=self.api_key)
        structlogger.debug("inkeep_document_retrieval._get_client", base_url=base_url)

        try:
            yield client
        except Exception as e:
            structlogger.error(
                "inkeep_document_retrieval.client_error",
                event_info="InKeep Document Retrieval: Client error",
                error=str(e),
            )
            raise

    def _parse_documents_from_response(
        self, response: ChatCompletion
    ) -> List[Document]:
        """Parse the InKeep AI response into Document objects.

        Args:
            response: ChatCompletion response from InKeep AI's RAG model.

        Returns:
            List of Document objects. Empty list is returned if the response is empty
            or if the response is invalid.
        """
        try:
            content = response.choices[0].message.content
            if not content:
                return []

            response_data = json.loads(content)
            documents = []

            for item in response_data.get("content", []):
                try:
                    document = Document.from_inkeep_rag_response(item)
                    documents.append(document)
                except Exception as e:
                    structlogger.warning(
                        "inkeep_document_retrieval.invalid_document_skipped",
                        event_info=(
                            "InKeep Document Retrieval: Invalid document structure "
                            "skipped. Returning empty list for this item."
                        ),
                        error=str(e),
                        item=item,
                    )
                    # Continue processing other items, skip this invalid one
                    continue

            return documents

        except json.JSONDecodeError as e:
            structlogger.warning(
                "inkeep_document_retrieval.parse_documents_from_response"
                ".parse_response_failed",
                event_info=(
                    "InKeep Document Retrieval: Parse response failed. "
                    "Returning empty list.",
                ),
                error=str(e),
            )
            return []
        except Exception as e:
            structlogger.error(
                "inkeep_document_retrieval.parse_documents_from_response.error",
                event_info=(
                    "InKeep Document Retrieval: Parse response error. "
                    "Returning empty list.",
                ),
                error=str(e),
            )
            return []
