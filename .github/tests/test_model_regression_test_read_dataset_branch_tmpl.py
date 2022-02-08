import pathlib
import subprocess
import pytest

TEMPLATE_FPATH = ".github/templates/model_regression_test_read_dataset_branch.tmpl"
REPO_DIR = pathlib.Path("").absolute()
TEST_DATA_DIR = str(pathlib.Path(__file__).parent / 'test_data')
DEFAULT_DATASET_BRANCH = "main"


@pytest.mark.parametrize(["comment_body_file", "expected_dataset_branch"],
                         [
                            ("comment_body.json", "test_dataset_branch"),
                            ("comment_body_no_dataset_branch.json", "main")
                         ])
def test_read_dataset_branch(comment_body_file, expected_dataset_branch):
    CMD = ("gomplate "
           f"-d github={TEST_DATA_DIR}/{comment_body_file} "
           f"-f {TEMPLATE_FPATH}")
    output = subprocess.check_output(CMD.split(' '), cwd=REPO_DIR)
    output = output.decode("utf-8").strip()
    assert output == f"export DATASET_BRANCH=\"{expected_dataset_branch}\""
