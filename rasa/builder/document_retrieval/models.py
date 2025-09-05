from typing import Any, Dict, Optional

import structlog
from pydantic import BaseModel, Field

structlogger = structlog.get_logger()


class Document(BaseModel):
    """Model for document retrieval results."""

    content: str = Field(...)
    url: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    @classmethod
    def from_inkeep_rag_response(cls, rag_item: Dict[str, Any]) -> "Document":
        """Create a Document object from a single InKeep RAG response item.

        Args:
            rag_item: Single item from InKeep RAG response

        Returns:
            Document object with extracted content and metadata
        """
        source = rag_item.get("source", {})
        text_content = cls._extract_text_from_source(source)

        return cls(
            content=text_content.strip() if text_content else "",
            url=rag_item.get("url"),
            title=rag_item.get("title"),
            metadata={
                "type": rag_item.get("type"),
                "record_type": rag_item.get("record_type"),
                "context": rag_item.get("context"),
                "media_type": source.get("media_type"),
            },
        )

    @staticmethod
    def _extract_text_from_source(source: Dict[str, Any]) -> str:
        """Extract text content from InKeep source object.

        Args:
            source: Source object from InKeep RAG response

        Returns:
            Extracted text content
        """
        # Try to extract from content array first
        if "content" in source:
            text_parts = []
            for content_item in source["content"]:
                if content_item.get("type") == "text" and content_item.get("text"):
                    text_parts.append(content_item["text"])
            if text_parts:
                return "\n".join(text_parts)

        # Fallback to source data
        return source.get("data", "")
