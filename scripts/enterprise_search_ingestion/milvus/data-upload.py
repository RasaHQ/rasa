# Usage: python data-upload.py --docs_folder data/documents

import logging
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from pathlib import Path
from langchain.schema import Document
from typing import List
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_documents(docs_folder: str) -> List[Document]:
    """Extract documents from a given folder.

    Args:
        docs_folder: The folder containing the documents.

    Returns:
        the list of documents
    """
    logger.debug(f"Extracting files from: {Path(docs_folder).absolute()}")
    # check if directory exists
    if not Path(docs_folder).exists():
        raise SystemExit(f"Directory '{docs_folder}' does not exist.")

    # change the loader_cls for a different document type or extension
    # https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
    loader = DirectoryLoader(
        docs_folder, glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    return loader.load()


def create_chunks(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Splits the documents into chunks with RecursiveCharacterTextSplitter.

    Args:
        documents: The documents to split.
        chunk_size: The size of the chunks.
        chunk_overlap: The overlap of the chunks.

    Returns:
        The list of chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def create_milvus_collection(
        docs: List[Document],
        host: str,
        port: str,
        user: str,
        password: str,
        collection_name: str,
) -> Milvus:
    """Creates a Milvus collection from the documents.

    Args:
        docs: The documents to store.
        host: The Milvus host.
        port: The Milvus port.
        user: The Milvus user.
        password: The Milvus password.
        collection: The Milvus collection name.

    Returns:
        The Milvus collection.
    """
    # change here if using a different embedding model
    embeddings = OpenAIEmbeddings()
    connection_args = {
        "host": host,
        "port": port,
    }

    if user:
        connection_args["user"] = user
    if password:
        connection_args["password"] = password

    return Milvus.from_documents(
        docs,
        embeddings,
        collection_name=collection_name,
        connection_args=connection_args,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="data-upload.py",
        description="Extract documents from a folder and create a Milvus collection. Requires the pymilvus package. Make sure that you have the OPENAI_API_KEY environment variables set.",
        epilog="Example: python data-upload.py --docs_folder data/documents",
    )
    parser.add_argument("--docs_folder", type=str, default="data/documents",
                        help="The folder containing the txt documents.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="The size of the chunks.")
    parser.add_argument("--chunk_overlap", type=int, default=20, help="The overlap of the chunks.")
    parser.add_argument("--milvus_host", type=str, default="localhost", help="The Milvus host.")
    parser.add_argument("--milvus_port", type=str, default="19530", help="The Milvus port.")
    parser.add_argument("--milvus_user", type=str, default=None, help="The Milvus user.")
    parser.add_argument("--milvus_pass", type=str, default=None, help="The Milvus password.")
    parser.add_argument("--milvus_collection", type=str, default="rasa", help="The Milvus collection name.")
    args = parser.parse_args()

    docs = extract_documents(args.docs_folder)
    logger.info(f"{len(docs)} documents extracted.")
    chunks = create_chunks(docs, args.chunk_size, args.chunk_overlap)
    logger.info(f"{len(chunks)} chunks created.")
    for i, chunk in enumerate(chunks[:3]):
        logger.info(f"chunk {i}")
        logger.info(chunk)
    create_milvus_collection(
        docs=chunks,
        host=args.milvus_host,
        port=args.milvus_port,
        user=args.milvus_user,
        password=args.milvus_pass,
        collection_name=args.milvus_collection,
    )
    logger.info("Milvus collection created.")


if __name__ == "__main__":
    main()
