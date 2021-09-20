import filecmp
from pathlib import Path

from yarl import URL


def latest_request(mocked, request_type, path):
    return mocked.requests.get((request_type, URL(path)))


def json_of_latest_request(r):
    return r[-1].kwargs["json"]


def are_directory_contents_equal(dir1: Path, dir2: Path) -> bool:
    """ Compare two directories recursively.

    Files in each directory are
    assumed to be equal if their names and contents are equal.

    Args:
        dir1: The first directory.
        dir2: The second directory.

    Returns:
        `True` if they are equal, `False` otherwise.
    """
    dirs_cmp = filecmp.dircmp(dir1, dir2)
    if dirs_cmp.left_only or dirs_cmp.right_only:
        return False

    (_, mismatches, errors) = filecmp.cmpfiles(
        dir1, dir2, dirs_cmp.common_files, shallow=False
    )

    if mismatches or errors:
        return False

    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = Path(dir1, common_dir)
        new_dir2 = Path(dir2, common_dir)

        is_equal = are_directory_contents_equal(new_dir1, new_dir2)
        if not is_equal:
            return False

    return True
