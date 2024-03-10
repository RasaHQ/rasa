import pathlib
import subprocess
import pytest
from typing import Text

TEMPLATE_FPATH = ".github/templates/model_regression_test_read_dataset_branch.tmpl"
REPO_DIR = pathlib.Path("").absolute()
TEST_DATA_DIR = str(pathlib.Path(__file__).parent / "test_data")
DEFAULT_DATASET_BRANCH = "main"


@pytest.mark.parametrize(
    "comment_body_file,expected_dataset_branch",
    [
        ("comment_body.json", "test_dataset_branch"),
        ("comment_body_no_dataset_branch.json", DEFAULT_DATASET_BRANCH),
    ],
)
def test_read_dataset_branch(comment_body_file: Text, expected_dataset_branch: Text):
    cmd = (
        "gomplate "
        f"-d github={TEST_DATA_DIR}/{comment_body_file} "
        f"-f {TEMPLATE_FPATH}"
    )
    output = subprocess.check_output(cmd.split(" "), cwd=REPO_DIR)
    output = output.decode("utf-8").strip()
    assert output == f'export DATASET_BRANCH="{expected_dataset_branch}"'
