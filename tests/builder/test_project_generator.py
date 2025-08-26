import io
import tarfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from rasa.builder.project_generator import _safe_tar_members


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
