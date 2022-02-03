import pathlib
import subprocess

TEMPLATE_FPATH = ".github/templates/model_regression_test_results.tmpl"
REPO_DIR = pathlib.Path("").absolute()
TEST_DATA_DIR = str(pathlib.Path(__file__).parent / 'test_data')


def test_comment_text():
    CMD = ("gomplate "
           f"-d data={TEST_DATA_DIR}/report_single_dictformat.json "
           f"-d results_main={TEST_DATA_DIR}/report-on-schedule-2022-01-10.json "
           f"-f {TEMPLATE_FPATH}")
    output = subprocess.check_output(CMD.split(' '), cwd=REPO_DIR)
    output = output.decode("utf-8")
    expected_output = """
Dataset: `RasaHQ/financial-demo`, Dataset repository branch: `fix-model-regression-tests` (external repository), commit: `52a3ad3eb5292d56542687e23b06703431f15ead`
Configuration repository branch: `main`
| Configuration | Intent Classification Micro F1 | Entity Recognition Micro F1 | Response Selection Micro F1 |
|---------------|-----------------|-----------------|-------------------|
| `BERT + DIET(seq) + ResponseSelector(t2t)`<br> test: `1m29s`, train: `2m55s`, total: `4m24s`|1.0000 (`no data`)|0.8333 (`no data`)|`no data`|


"""
    assert output == expected_output
