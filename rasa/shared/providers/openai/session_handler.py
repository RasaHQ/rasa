import ssl
from pathlib import Path
from ssl import SSLContext
from typing import Text

import certifi
import openai
import structlog
from aiohttp import ClientSession, TCPConnector

structlogger = structlog.get_logger()


class OpenAISessionHandler:
    """
    This context manager is used to manage the aiohttp session for OpenAI. This
    session handles calls that use self-signed certificates, the path to which
    must be provided.

    Each session is unique to the context in which it is used, ensuring thread
    safety across asynchronous tasks.

    Upon entering the context manager, the session is configured with a custom
    SSL context. Upon exiting, the session is properly closed and the context
    variable is cleared.
    """

    def __init__(
        self,
        certificate_path: Text = certifi.where(),
        purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
    ):
        self._certificate_path = certificate_path
        self._purpose = purpose

    async def __aenter__(self) -> ClientSession:
        session = openai.aiosession.get()
        if session is None:
            session = self._create_session()
            openai.aiosession.set(session)
            structlogger.debug(
                "openai_aiohttp_session_handler" ".set_openai_session",
            )
        return session

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        session = openai.aiosession.get()
        if session:
            await session.close()
            structlogger.debug(
                "openai_aiohttp_session_handler" ".close_openai_session",
            )
            openai.aiosession.set(None)
            structlogger.debug(
                "openai_aiohttp_session_handler" ".clear_openai_session",
            )

    def _create_session(self) -> ClientSession:
        """
        Create client session with an SSL context
        created from the certificate path
        """
        ssl_context = self._create_ssl_context()
        conn = TCPConnector(ssl=ssl_context)
        session = ClientSession(connector=conn)
        return session

    def _create_ssl_context(self) -> SSLContext:
        """
        Create an SSL context using the provided certificate file path.

        Returns:
            SSLContext: An SSL context configured with the given certificate.

        Raises:
            ValueError: If the certificate path is invalid.
            Exception: If there is an error creating the SSL context.
        """
        if self._certificate_path is None or not Path(self._certificate_path).is_file():
            structlogger.error(
                "openai_aiohttp_session_handler"
                ".create_ssl_context"
                ".cannot_load_certificate_from_path",
                event_info=(
                    f"Cannot load certificate file from the "
                    f"given path: {self._certificate_path}"
                ),
            )
            raise ValueError("Invalid certificate file path.")
        try:
            ssl_context = ssl.create_default_context(
                purpose=self._purpose, cafile=self._certificate_path
            )
            purpose = (
                "SERVER_AUTH"
                if self._purpose == ssl.Purpose.SERVER_AUTH
                else "CLIENT_AUTH"
            )
            structlogger.debug(
                "openai_aiohttp_session_handler" ".created_ssl_context" ".created",
                certificate_file=self._certificate_path,
                purpose=f"{purpose} - {self._purpose}",
            )
            return ssl_context
        except Exception as e:
            structlogger.error(
                "openai_aiohttp_session_handler" ".create_ssl_context" ".error",
                error=e,
            )
            raise
