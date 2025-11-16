"""Download utilities for bot projects."""

import asyncio
import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Dict, Optional, Union
from urllib.parse import urlparse

import aiofiles
import aiohttp

from rasa.builder.config import COPILOT_DB_RELATIVE_PATH
from rasa.builder.constants import MAX_BACKUP_SIZE
from rasa.builder.exceptions import ProjectGenerationError


def _get_env_content() -> str:
    """Generate .env file content."""
    return f"RASA_PRO_LICENSE={os.getenv('RASA_PRO_LICENSE')}\n"


def _get_python_version_content() -> str:
    """Generate .python-version file content with current Python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}\n"


def _get_pyproject_toml_content(project_id: str) -> str:
    """Generate pyproject.toml file content."""
    return dedent(
        f"""
        [project]
        name = "{project_id}"
        version = "0.1.0"
        description = "Add your description for your Rasa bot here"
        readme = "README.md"
        dependencies = ["rasa-pro>=3.14"]
        requires-python = ">={sys.version_info.major}.{sys.version_info.minor}"
        """
    )


def _get_readme_content() -> str:
    """Generate README.md file content with basic instructions."""
    return dedent(
        """
        # Rasa Assistant

        This is your Rasa assistant project. Below are the basic commands to
        get started.

        ## Prerequisites
        Make sure you have [uv](https://docs.astral.sh/uv/) installed.

        ## Training the Assistant

        To train your assistant with the current configuration and training data:

        ```bash
        uv run rasa train
        ```

        This will create a model in the `models/` directory.

        ## Testing the Assistant

        To test your assistant interactively in the command line:

        ```bash
        uv run rasa inspect
        ```

        ## Running the Assistant

        To start the Rasa server:

        ```bash
        uv run rasa run
        ```

        The server will be available at `http://localhost:5005`.

        ## Project Structure

        - `config.yml` - Configuration for your NLU pipeline and policies
        - `domain.yml` - Defines intents, entities, slots, responses, and actions
        - `data/` - Flows of your bot
        - `actions/` - Custom action code (if any)

        ## Next Steps

        1. Customize your domain and flows
        2. Train your model with `rasa train`
        3. Test your assistant with `rasa inspect`
        4. Deploy your assistant with `rasa run`

        For more information, visit the [Rasa documentation](https://rasa.com/docs/).
        """
    )


def _add_file_to_tar(
    tar: tarfile.TarFile, filename: str, content: Union[str, bytes]
) -> None:
    """Add a file with the given content to the tar archive.

    Args:
        tar: The tar file object to add to
        filename: Name of the file in the archive
        content: Content of the file as a string or bytes
    """
    file_data = content if isinstance(content, bytes) else content.encode("utf-8")
    tarinfo = tarfile.TarInfo(name=filename)
    tarinfo.size = len(file_data)
    tar.addfile(tarinfo, io.BytesIO(file_data))


def create_bot_project_archive(
    bot_files: Dict[str, Optional[str]], project_id: str, project_folder: Path
) -> bytes:
    """Create a tar.gz archive containing bot files and additional project files.

    Args:
        bot_files: Dictionary mapping file names to their content
        project_id: Name of the project for the archive filename and pyproject.toml
        project_folder: Path to the project folder

    Returns:
        bytes: The tar.gz archive data
    """
    tar_buffer = io.BytesIO()

    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        # Add bot files to archive
        for filename, content in bot_files.items():
            if content is not None:
                _add_file_to_tar(tar, filename, content)

        # Add copilot database if it exists
        copilot_db_path = project_folder / COPILOT_DB_RELATIVE_PATH
        if copilot_db_path.exists():
            with open(copilot_db_path, "rb") as db_file:
                _add_file_to_tar(tar, COPILOT_DB_RELATIVE_PATH, db_file.read())

        # Add additional project files
        _add_file_to_tar(tar, ".env", _get_env_content())
        _add_file_to_tar(tar, ".python-version", _get_python_version_content())
        _add_file_to_tar(
            tar,
            "pyproject.toml",
            _get_pyproject_toml_content(project_id),
        )
        _add_file_to_tar(tar, "README.md", _get_readme_content())

    tar_buffer.seek(0)
    return tar_buffer.getvalue()


def validate_s3_url(url: str) -> None:
    """Validate that the URL is from an expected S3 domain for security.

    Args:
        url: The URL to validate

    Raises:
        ValueError: If the URL is not from an expected S3 domain
    """
    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        raise ValueError("URL must have a valid hostname")

    hostname = hostname.lower()
    if not ("s3" in hostname and hostname.endswith(".amazonaws.com")):
        raise ValueError(f"URL must be from an AWS S3 domain, got: {hostname}")


async def download_backup_from_url(url: str) -> str:
    """Download backup file from presigned URL to a temporary file.

    Args:
        url: Presigned URL to download from

    Returns:
        Path to the downloaded temporary file

    Raises:
        ProjectGenerationError: If download fails or file is too large
    """
    # Validate URL for security
    validate_s3_url(url)

    # Create temporary file path (using mktemp for path only, not creating the file)
    temp_file_fd, temp_file_path = tempfile.mkstemp(suffix=".tar.gz")
    os.close(temp_file_fd)  # Close the file descriptor immediately

    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ProjectGenerationError(
                        f"Failed to download backup from presigned URL. "
                        f"HTTP {response.status}: {response.reason}",
                        attempts=1,
                    )

                # Check content length if available
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_BACKUP_SIZE:
                    raise ProjectGenerationError(
                        f"Backup file too large "
                        f"({content_length} bytes > {MAX_BACKUP_SIZE} bytes). "
                        f"Please provide a smaller backup file.",
                        attempts=1,
                    )

                # Stream download to file using async file operations
                downloaded_size = 0
                async with aiofiles.open(temp_file_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        downloaded_size += len(chunk)

                        # Check size limit during download
                        if downloaded_size > MAX_BACKUP_SIZE:
                            raise ProjectGenerationError(
                                f"Backup file too large "
                                f"({downloaded_size} bytes > {MAX_BACKUP_SIZE} bytes).",
                                attempts=1,
                            )

                        await f.write(chunk)

                return temp_file_path

    except ProjectGenerationError:
        # Clean up temp file and re-raise ProjectGenerationError as-is
        try:
            Path(temp_file_path).unlink(missing_ok=True)
        except Exception:
            pass
        raise
    except asyncio.TimeoutError:
        error_message = "Download timeout: Presigned URL may have expired."
    except aiohttp.ClientError as exc:
        error_message = f"Network error downloading backup: {exc}"
    except Exception as exc:
        error_message = f"Unexpected error downloading backup: {exc}"

    # Clean up temp file and raise error
    try:
        Path(temp_file_path).unlink(missing_ok=True)
    except Exception:
        pass
    raise ProjectGenerationError(error_message, attempts=1)
