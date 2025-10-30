import asyncio
import functools
from typing import Any, Dict, List, Optional

import numpy as np
import structlog

from rasa.shared.constants import (
    HUGGINGFACE_LOCAL_EMBEDDING_CACHING_FOLDER,
)
from rasa.shared.providers._configs.huggingface_local_embedding_client_config import (
    HuggingFaceLocalEmbeddingClientConfig,
)
from rasa.shared.providers.embedding.embedding_response import EmbeddingResponse

structlogger = structlog.get_logger()


class HuggingFaceLocalEmbeddingClient:
    """This client facilitates the computation of embeddings using locally
    stored HuggingFace transformer models. It utilizes the
    `sentence-transformers` package.

    To use, you should have the `sentence_transformers` python package
    installed.

    Parameters:
        model (str): Identifier for the model or path to a directory containing
            model files.
        cache_folder (Optional[str]): Path to the folder for caching models.
            Default is defined by `HUGGINGFACE_LOCAL_EMBEDDING_CACHING_FOLDER`.
        model_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments
            for model configuration.
        encode_kwargs (Optional[Dict[str, Any]]): Parameters to control the
            encoding process.
        multi_process (bool): Flag to enable or disable multiprocessing for
            embedding generation.
        show_progress (bool): Whether to show a progress bar during embedding
            generation.
    """

    def __init__(
        self,
        model: str,
        cache_folder: Optional[str] = str(HUGGINGFACE_LOCAL_EMBEDDING_CACHING_FOLDER),
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        multi_process: bool = False,
        show_progress: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._model_name_or_path = model
        self._multi_process = multi_process
        self._cache_folder = cache_folder
        self._model_kwargs = model_kwargs or {}
        self._encode_kwargs = encode_kwargs or {}
        self._show_progress = show_progress
        self._init_client()

    def _init_client(self) -> None:
        self._validate_if_sentence_transformers_installed()

        import sentence_transformers

        self._client = sentence_transformers.SentenceTransformer(
            model_name_or_path=self._model_name_or_path,
            cache_folder=self._cache_folder,
            **self._model_kwargs,
        )

    @property
    def model(self) -> str:
        return self._model_name_or_path

    @property
    def config(self) -> Dict:
        config = HuggingFaceLocalEmbeddingClientConfig(
            model=self._model_name_or_path,
            cache_folder=self._cache_folder,
            show_progress=self._show_progress,
            model_kwargs=self._model_kwargs,
            encode_kwargs=self._encode_kwargs,
            multi_process=self._multi_process,
        )
        return config.to_dict()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HuggingFaceLocalEmbeddingClient":
        huggingface_config = HuggingFaceLocalEmbeddingClientConfig.from_dict(config)
        return cls(
            model=huggingface_config.model,
            cache_folder=huggingface_config.cache_folder,
            model_kwargs=huggingface_config.model_kwargs,
            encode_kwargs=huggingface_config.encode_kwargs,
            show_progress=huggingface_config.show_progress,
        )

    @staticmethod
    def _validate_if_sentence_transformers_installed() -> None:
        """In order to use the local embeddings from HuggingFace,
        sentence-transformer must be installed.

        Raises:
            ImportError: If 'sentence-transformers' is not installed.
        """
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            message = (
                "Could not import sentence_transformers python package. "
                "Please install it with 'pip install sentence-transformers'."
            )
            structlogger.error(
                "huggingface_local_embedding_client.sentence_transformers_not_installed",
                message=message,
            )
            raise ImportError(message) from exc

    def embed(self, documents: List[str], **kwargs: Any) -> EmbeddingResponse:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            documents: The list of documents to embed.

        Returns:
            List of embeddings, one for each text.
        """
        self._validate_if_sentence_transformers_installed()
        documents = list(map(lambda x: x.replace("\n", " "), documents))

        if self._multi_process:
            embeddings = self._embed_with_multiprocessing(documents)
        else:
            embeddings = self._embed_without_multiprocessing(documents)

        return EmbeddingResponse(
            data=embeddings.tolist(),
            model=self._model_name_or_path,
        )

    def _embed_without_multiprocessing(self, documents: List[str]) -> np.ndarray:
        embeddings: np.ndarray = self._client.encode(
            documents,
            show_progress_bar=self._show_progress,
            convert_to_numpy=True,
            **self._encode_kwargs,
        )
        return embeddings

    def _embed_with_multiprocessing(self, documents: List[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        pool = self._client.start_multi_process_pool()
        embeddings = self._client.encode_multi_process(documents, pool)
        SentenceTransformer.stop_multi_process_pool(pool)
        return embeddings

    async def aembed(self, documents: List[str], **kwargs: Any) -> EmbeddingResponse:
        """Asynchronous Embed search docs.

        Args:
            documents: List of documents to embed.

        Returns:
            List of embeddings.
        """
        loop = asyncio.get_running_loop()
        # Using run_in_executor to execute the synchronous embedding function
        # in a separate thread
        embeddings = await loop.run_in_executor(
            None, functools.partial(self.embed, documents, **kwargs)
        )
        return embeddings

    def validate_documents(self, documents: List[str]) -> None:
        for doc in documents:
            if not isinstance(doc, str):
                message = "All documents must be strings."
                structlogger.error(
                    message=message,
                    doc=doc,
                )
                raise ValueError(message)
            if not doc.strip():
                message = "Documents cannot be empty or whitespace."
                structlogger.error(
                    message=message,
                    doc=doc,
                )
                raise ValueError(message)

    def validate_client_setup(self, *args, **kwargs) -> None:  # type: ignore
        pass
