import io
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from unittest.mock import patch

import pytest

from rasa.builder.project_generator import ProjectGenerator
from rasa.builder.template_cache import (
    _safe_tar_members,
    download_cache_for_template,
)
from rasa.cli.scaffold import ProjectTemplateName


def _build_tar(entries: Iterable[Tuple[str, str, Optional[str]]]) -> tarfile.TarFile:
    """Create an in-memory tar.gz with given entries.

    Each entry is a tuple of (name, type, link_target). Supported types: file, dir,
    symlink, hardlink. link_target is used for link types.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, entry_type, link_target in entries:
            info = tarfile.TarInfo(name=name)
            if entry_type == "dir":
                info.type = tarfile.DIRTYPE
                info.size = 0
                tar.addfile(info)
            elif entry_type == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = link_target or "target"
                info.size = 0
                tar.addfile(info)
            elif entry_type == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = link_target or "target"
                info.size = 0
                tar.addfile(info)
            else:
                # regular file
                info.size = 0
                tar.addfile(info)

    buffer.seek(0)
    return tarfile.open(fileobj=buffer, mode="r:gz")


def _member_names(members: Iterable[tarfile.TarInfo]) -> List[str]:
    return [m.name for m in members]


def test_safe_tar_members_filters_traversal_and_links(tmp_path: Path) -> None:
    tar = _build_tar(
        [
            ("ok.txt", "file", None),
            ("dir/sub.txt", "file", None),
            ("../evil.txt", "file", None),
            ("/abs.txt", "file", None),
            ("dir/../../escape.txt", "file", None),
            ("link", "symlink", "ok.txt"),
            ("hard", "hardlink", "ok.txt"),
            ("dir/", "dir", None),
        ]
    )

    try:
        safe = list(_safe_tar_members(tar, tmp_path))
        names = _member_names(safe)

        assert "ok.txt" in names
        assert "dir/sub.txt" in names
        # directory entries may be normalized without trailing slash in some tar impls
        assert any(n in ("dir", "dir/") for n in names)

        assert "../evil.txt" not in names
        assert "/abs.txt" not in names
        assert "dir/../../escape.txt" not in names
        assert "link" not in names
        assert "hard" not in names
    finally:
        tar.close()


def test_safe_tar_members_allows_normalized_paths(tmp_path: Path) -> None:
    tar = _build_tar(
        [
            ("dir/../ok2.txt", "file", None),
            ("a/./b.txt", "file", None),
        ]
    )

    try:
        safe = list(_safe_tar_members(tar, tmp_path))
        names = _member_names(safe)

        # Both entries resolve within base directory and should be allowed
        assert "dir/../ok2.txt" in names
        assert "a/./b.txt" in names
    finally:
        tar.close()


class TestProjectGenerator:
    """Test ProjectGenerator class methods."""

    def test_is_empty_with_empty_directory(self, tmp_path: Path) -> None:
        """Test is_empty returns True for empty directory."""
        generator = ProjectGenerator(tmp_path)
        assert generator.is_empty() is True

    def test_is_empty_with_files(self, tmp_path: Path) -> None:
        """Test is_empty returns False when directory contains files."""
        (tmp_path / "config.yml").write_text("version: '3.1'")
        generator = ProjectGenerator(tmp_path)
        assert generator.is_empty() is False

    def test_is_empty_ignores_hidden_files(self, tmp_path: Path) -> None:
        """Test is_empty ignores hidden files and directories."""
        (tmp_path / ".hidden_file").write_text("hidden")
        (tmp_path / ".hidden_dir").mkdir()
        generator = ProjectGenerator(tmp_path)
        assert generator.is_empty() is True

    def test_is_empty_with_empty_subdirectories(self, tmp_path: Path) -> None:
        """Test is_empty returns True when directory contains only empty subdirs."""
        (tmp_path / "data").mkdir()
        generator = ProjectGenerator(tmp_path)
        # The implementation only checks for files, not directories
        assert generator.is_empty() is True

    def test_is_empty_with_files_in_subdirectories(self, tmp_path: Path) -> None:
        """Test is_empty returns True when files are only in subdirs."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file.txt").write_text("content")
        generator = ProjectGenerator(tmp_path)
        # Should return True because is_empty only checks for files at the root level
        assert generator.is_empty() is True

    @pytest.mark.asyncio
    async def test_init_from_template_uses_copy_cache_instead_of_download(
        self, tmp_path: Path
    ) -> None:
        """Test that init_from_template uses copy_cache instead of downloading."""
        generator = ProjectGenerator(tmp_path)

        with (
            patch(
                "rasa.builder.project_generator.create_initial_project"
            ) as mock_create,
            patch(
                "rasa.builder.project_generator.copy_cache_for_template_if_available"
            ) as mock_copy_cache,
            patch("rasa.builder.project_generator.ensure_first_used") as mock_ensure,
        ):
            await generator.init_from_template(ProjectTemplateName.DEFAULT)

            # Verify the correct sequence of calls
            mock_create.assert_called_once_with(
                tmp_path.as_posix(), ProjectTemplateName.DEFAULT
            )
            mock_copy_cache.assert_called_once_with(
                ProjectTemplateName.DEFAULT, tmp_path
            )
            mock_ensure.assert_called_once_with(tmp_path)


class TestDownloadCacheForTemplate:
    """Test the download_cache_for_template function from project_generator.py."""

    @pytest.fixture
    def sample_tar_content(self, tmp_path: Path) -> bytes:
        """Create a sample tar.gz file content for testing."""
        # Create a temporary directory with some files
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        (source_dir / "config.yml").write_text("version: '3.1'")
        (source_dir / "domain.yml").write_text("version: '3.1'")
        (source_dir / ".rasa").mkdir()
        (source_dir / ".rasa" / "model.tar.gz").write_text("model data")

        # Create tar.gz
        tar_path = tmp_path / "sample.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(source_dir, arcname=".")

        return tar_path.read_bytes()

    @pytest.mark.asyncio
    async def test_temporary_file_cleanup(self, tmp_path: Path) -> None:
        """Test that temporary files are cleaned up even on error."""
        target_dir = tmp_path / "target"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.side_effect = Exception("Test error")

            # Count temp files before
            temp_files_before = len(list(Path(tempfile.gettempdir()).glob("*.tar.gz")))

            await download_cache_for_template(
                ProjectTemplateName.DEFAULT, str(target_dir)
            )

            # Count temp files after - should be the same
            temp_files_after = len(list(Path(tempfile.gettempdir()).glob("*.tar.gz")))
            assert temp_files_after == temp_files_before
