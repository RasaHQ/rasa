"""Tests for MCP tools utility helpers."""

from pathlib import Path

import pytest

from rasa.builder.copilot.mcp_server.tools.utils import validate_subfolder_path
from rasa.utils.io import InvalidPathException


class TestValidateSubfolderPath:
    """Test validate_subfolder_path rejects path traversal attacks."""

    @pytest.mark.parametrize(
        "subfolder",
        ["data", "a/b", "future_dir", "."],
        ids=["simple", "nested", "nonexistent", "dot"],
    )
    def test_valid_subfolder_is_accepted(self, tmp_path: Path, subfolder: str) -> None:
        """Subfolders that stay within the project root are accepted."""
        result = validate_subfolder_path(tmp_path, subfolder)
        assert result == (tmp_path / subfolder).resolve()

    @pytest.mark.parametrize(
        "subfolder",
        [
            "/etc",
            "/tmp/",
            "../secret",
            "../../etc",
            "data/../../etc",
            "/".join([".."] * 50),
        ],
        ids=[
            "absolute",
            "absolute-trailing",
            "single-dotdot",
            "double-dotdot",
            "hidden-in-middle",
            "escape-to-root",
        ],
    )
    def test_traversal_is_rejected(self, tmp_path: Path, subfolder: str) -> None:
        """Paths that escape the project root raise InvalidPathException."""
        with pytest.raises(InvalidPathException, match="Path traversal detected"):
            validate_subfolder_path(tmp_path, subfolder)
