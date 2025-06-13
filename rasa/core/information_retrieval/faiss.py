from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Text

import structlog
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores.faiss import FAISS

from rasa.core.information_retrieval import (
    InformationRetrieval,
    InformationRetrievalException,
    SearchResultList,
)
from rasa.core.information_retrieval.ingestion.faq_parser import _format_faq_documents
from rasa.utils.endpoints import EndpointConfig
from rasa.utils.ml_utils import persist_faiss_vector_store

if TYPE_CHECKING:
    from langchain.schema import Document
    from langchain.schema.embeddings import Embeddings

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
        if parsed_documents:
            texts = [document.page_content for document in parsed_documents]
            metadatas = [document.metadata for document in parsed_documents]
            return FAISS.from_texts(texts, embedding, metadatas=metadatas, ids=None)
        else:
            raise ValueError(f"No documents found at '{docs_folder}'.")

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
