"""MCP tool for searching Rasa documentation."""

from typing import List

import structlog

from rasa.builder.copilot.mcp_server.models import (
    DocumentSearchResponse,
    DocumentSearchResult,
)
from rasa.builder.document_retrieval.inkeep_document_retrieval import (
    InKeepDocumentRetrieval,
)
from rasa.builder.document_retrieval.models import Document

structlogger = structlog.get_logger()


async def search_rasa_documentation(query: str) -> DocumentSearchResponse:
    """Search Rasa documentation for relevant information.

    Args:
        query: The search query to find relevant documentation

    Returns:
        DocumentSearchResponse with search results (title, url, content).
        Returns empty list if no results.
    """
    try:
        document_retrieval = InKeepDocumentRetrieval()
        documents: List[Document] = await document_retrieval.retrieve_documents(query)

        # Format documents as structured results
        results = [
            DocumentSearchResult(
                index=idx,
                title=doc.title,
                url=doc.url,
                content=doc.content,
            )
            for idx, doc in enumerate(documents, start=1)
        ]

        return DocumentSearchResponse(documents=results)

    except Exception as e:
        structlogger.error(
            "mcp_server.tools.document_search.error",
            event_info="MCP tool failed to search documentation",
            query=query,
            error=str(e),
        )
        return DocumentSearchResponse(
            documents=[],
            error=f"Failed to search documentation: {e!s}",
        )
