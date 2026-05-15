from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Text

import structlog
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rasa.core.information_retrieval import (
    InformationRetrieval,
    InformationRetrievalException,
    SearchResultList,
)
from rasa.core.information_retrieval.ingestion.faq_parser import _format_faq_documents
from rasa.utils.endpoints import EndpointConfig
from rasa.utils.ml_utils import persist_faiss_vector_store

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

logger = structlog.get_logger()


class FAISS_Store(InformationRetrieval):
    """FAISS Store implementation."""

    def __init__(
        self,
        embeddings: "Embeddings",
        index_path: str,
        docs_folder: Optional[str],
        create_index: Optional[bool] = False,
        parse_as_faq_pairs: Optional[bool] = False,
    ):
        """Initializes the FAISS Store."""
        self.chunk_size = 1000
        self.chunk_overlap = 20
        self.parse_as_faq_pairs = parse_as_faq_pairs

        path = Path(index_path) / "documents_faiss"
        if create_index:
            logger.info(
                "information_retrieval.faiss_store.create_index", path=path.absolute()
            )
            self.index = self._create_document_index(docs_folder, embeddings)
            self._persist(path)
        else:
            logger.info(
                "information_retrieval.faiss_store.load_index", path=path.absolute()
            )
            self.index = FAISS.load_local(
                str(path), embeddings, allow_dangerous_deserialization=True
            )

    @staticmethod
    def load_documents(docs_folder: str) -> List["Document"]:
        """Loads documents from a given folder.

        Args:
            docs_folder: The folder containing the documents.

        Returns:
            the list of documents
        """
        logger.info(
            "information_retrieval.faiss_store.load_documents",
            docs_folder=Path(docs_folder).absolute(),
        )
        loader = DirectoryLoader(
            docs_folder, glob="**/*.txt", loader_cls=TextLoader, show_progress=True
        )

        return loader.load()

    def _create_document_index(
        self, docs_folder: Optional[str], embedding: "Embeddings"
    ) -> FAISS:
        """Creates a document index from the documents in the given folder.
        Args:
            docs_folder: The folder containing the documents.
            embedding: The embedding to use.
        Returns:
            The document index.
        """
        if not docs_folder:
            raise ValueError("parameter `docs_folder` needs to be specified")

        documents = self.load_documents(docs_folder)

        if not self.parse_as_faq_pairs:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )
            parsed_documents = splitter.split_documents(documents)
        else:
            parsed_documents = _format_faq_documents(documents)

        logger.info(
            "information_retrieval.faiss_store._create_document_index",
            len_chunks=len(parsed_documents),
        )

        if not parsed_documents:
            raise ValueError(f"No documents found at '{docs_folder}'.")

        texts = [document.page_content for document in parsed_documents]
        metadatas = [document.metadata for document in parsed_documents]

        # Warn if document size exceeds 2GB - FAISS in-memory performance degrades
        ONE_GB = 1024 * 1024 * 1024
        total_size_bytes = sum(len(text.encode("utf-8")) for text in texts)
        size_limit_bytes = 2 * ONE_GB  # 2GB
        if total_size_bytes > size_limit_bytes:
            logger.warning(
                "information_retrieval.faiss_store.large_document_size",
                message="Document size exceeds 2GB. FAISS in-memory vector store "
                "performance may degrade significantly with large datasets. "
                "Consider using a dedicated vector database for better performance.",
                total_size_gb=round(total_size_bytes / ONE_GB, 2),
            )

        # Batch size to avoid hitting embedding API token limits
        # Use a conservative batch size for large documents
        batch_size = 50
        total_batches = (len(texts) + batch_size - 1) // batch_size

        # Create index with first batch
        logger.info(
            "information_retrieval.faiss_store._create_document_index.embedding_batch",
            batch=1,
            total_batches=total_batches,
        )
        index = FAISS.from_texts(
            texts[:batch_size], embedding, metadatas=metadatas[:batch_size]
        )

        # Add remaining documents in batches
        for i in range(batch_size, len(texts), batch_size):
            batch_num = (i // batch_size) + 1
            logger.info(
                "information_retrieval.faiss_store._create_document_index.embedding_batch",
                batch=batch_num,
                total_batches=total_batches,
            )
            batch_texts = texts[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]
            index.add_texts(batch_texts, metadatas=batch_metadatas)

        return index

    def _persist(self, path: Path) -> None:
        persist_faiss_vector_store(path, self.index)

    def connect(self, config: EndpointConfig) -> None:
        """Faiss does not need to connect to a server."""
        pass

    async def search(
        self, query: Text, tracker_state: Dict[str, Any], threshold: float = 0.0
    ) -> SearchResultList:
        logger.debug("information_retrieval.faiss_store.search", query=query)
        try:
            documents = await self.index.as_retriever().ainvoke(query)
        except Exception as exc:
            raise InformationRetrievalException from exc

        return SearchResultList.from_document_list(documents)
