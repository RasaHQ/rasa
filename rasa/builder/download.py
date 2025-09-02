"""Download utilities for bot projects."""

import io
import os
import sys
import tarfile
from textwrap import dedent
from typing import Dict, Optional


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
        dependencies = ["rasa-pro>=3.13"]
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


def _add_file_to_tar(tar: tarfile.TarFile, filename: str, content: str) -> None:
    """Add a file with the given content to the tar archive.

    Args:
        tar: The tar file object to add to
        filename: Name of the file in the archive
        content: Content of the file as a string
    """
    file_data = content.encode("utf-8")
    tarinfo = tarfile.TarInfo(name=filename)
    tarinfo.size = len(file_data)
    tar.addfile(tarinfo, io.BytesIO(file_data))


def create_bot_project_archive(
    bot_files: Dict[str, Optional[str]], project_id: str
) -> bytes:
    """Create a tar.gz archive containing bot files and additional project files.

    Args:
        bot_files: Dictionary mapping file names to their content
        project_id: Name of the project for the archive filename and pyproject.toml

    Returns:
        bytes: The tar.gz archive data
    """
    tar_buffer = io.BytesIO()

    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        # Add bot files to archive
        for filename, content in bot_files.items():
            if content is not None:
                _add_file_to_tar(tar, filename, content)

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
