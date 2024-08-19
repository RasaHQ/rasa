import logging

from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores.milvus import Milvus
from langchain_community.vectorstores.qdrant import Qdrant
from typing import List
import yaml
from argparse import ArgumentParser
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "rasa"


def extract_all_webpages(url: str) -> List[Document]:
    """Extract webpages from a given url."""
    if url is None:
        raise SystemExit("URL not found in config file.")

    logger.info(f"Extracting URLs from: {url}")
    
    # send a request like a browser
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}

    # nothing to exclude
    exclude_list = []

    # how deep to go
    max_depth = 2

    loader = RecursiveUrlLoader(
        url=url,
        headers=headers,
        exclude_dirs=exclude_list, 
        max_depth=max_depth, 
        extractor=lambda x: BeautifulSoup(x, "html.parser").text
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


def embeddings_factory(type: str, config: dict = None):
    if type.lower() == "openai":
        return OpenAIEmbeddings(**config)
    if type.lower() == "huggingface":
        return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def create_milvus_collection(
        embeddings: Embeddings,
        docs: List[Document], 
        connection_args: dict,
    ) -> Milvus:
    """Creates a Milvus collection from the documents.

    Args:
        embeddings: embeddings model object
        docs: The documents to store.
        connection_args: The connection arguments.

    Returns:
        The Milvus collection.
    """
    host = connection_args["host"]
    port = connection_args["port"]
    user = connection_args.get("user")
    password = connection_args.get("password")
    collection_name = connection_args.get("collection", DEFAULT_COLLECTION_NAME)

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


def create_qdrant_collection(
        embeddings: Embeddings,
        docs: List[Document], 
        connection_args: dict,
    ) -> Qdrant:
    """Creates a Qdrant collection from the documents.

    Args:
        embeddings: embeddings model object
        docs: The documents to store as a List of document chunks
        connection_args: The connection arguments.

    Returns:
        The Qdrant collection.
    """
    host = connection_args.get("host", None)
    port = connection_args.get("port", 6333)
    path = connection_args.get("path", None)
    collection_name = connection_args.get("collection", DEFAULT_COLLECTION_NAME)

    return Qdrant.from_documents(
        docs,
        embeddings,
        host=host,
        port=port,
        collection_name=collection_name,
        path=path,
    )


def validate_destination(destination: str):
    if destination.lower() not in ["milvus", "qdrant"]:
        raise SystemExit(f"Destination '{destination}' not supported.")


def validate_embeddings_type(embeddings_type: str):
    if embeddings_type.lower() not in ["openai", "huggingface"]:
        raise SystemExit(f"Embeddings type '{embeddings_type}' not supported.")
    elif embeddings_type.lower() == "openai":
        # check if OPENAI_API_KEY is set
        if not "OPENAI_API_KEY" in os.environ:
            raise SystemExit("OPENAI_API_KEY environment variable not set.")


def main():
    parser = ArgumentParser(
        prog="ingest.py",
        description="Extract web pages recursively and load them into a vector store.",
        epilog="Example: python ingest_web.py --config config.yaml",
    )
    parser.add_argument('-c', '--config', required=True, help='config file path')
    args = parser.parse_args()
    opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(vars(args))

    # parse config file
    docs_folder = opt.get("url")
    chunk_size = opt.get("chunk_size", 1000)
    chunk_overlap = opt.get("chunk_overlap", 20)
    embeddings_type = opt.get("embeddings", "openai")
    destination = opt.get("destination")

    # do some validation
    try:
        connection_args = opt["connection_args"]
    except KeyError:
        raise SystemExit("connection_args not found in config file.")

    validate_destination(destination)
    validate_embeddings_type(embeddings_type)
    
    # extract all webpages
    docs = extract_all_webpages(docs_folder)
    logger.info(f"{len(docs)} webpages extracted.")
    
    # create chunks
    chunks = create_chunks(docs, chunk_size, chunk_overlap)
    logger.info(f"{len(chunks)} chunks created.")
    for i, chunk in enumerate(chunks[:3]):
        logger.info(f"chunk {i}")
        logger.info(chunk)
    
    # create embeddings
    embeddings = embeddings_factory(embeddings_type, {})
        
    # create milvus collection
    if destination.lower() == "milvus":
        create_milvus_collection(
            embeddings=embeddings,
            docs=chunks, 
            connection_args=connection_args,
        )
        logger.info(f"Milvus collection created with arguments {connection_args}")
    elif destination.lower() == "qdrant":
        create_qdrant_collection(
            embeddings=embeddings,
            docs=chunks, 
            connection_args=connection_args,
        )
        logger.info(f"Qdrant collection created with arguments {connection_args}")
    else:
        raise SystemExit(f"Destination '{destination}' not supported. Only (milvus, qdrant) are supported.")


if __name__ == "__main__":
    main()
