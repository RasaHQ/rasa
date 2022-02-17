import pathlib
import subprocess

TEMPLATE_FPATH = ".github/templates/model_regression_test_results.tmpl"
REPO_DIR = pathlib.Path("").absolute()
TEST_DATA_DIR = str(pathlib.Path(__file__).parent / "test_data")


def test_comment_nlu():
    cmd = (
        "gomplate "
        f"-d data={TEST_DATA_DIR}/report_listformat_nlu.json "
        f"-d results_main={TEST_DATA_DIR}/report-on-schedule-2022-02-02.json "
        f"-f {TEMPLATE_FPATH}"
    )
    output = subprocess.check_output(cmd.split(" "), cwd=REPO_DIR)
    output = output.decode("utf-8")
    expected_output = """
Dataset: `RasaHQ/financial-demo`, Dataset repository branch: `fix-model-regression-tests` (external repository), commit: `52a3ad3eb5292d56542687e23b06703431f15ead`
Configuration repository branch: `main`
| Configuration | Intent Classification Micro F1 | Entity Recognition Micro F1 | Response Selection Micro F1 |
|---------------|-----------------|-----------------|-------------------|
| `BERT + DIET(seq) + ResponseSelector(t2t)`<br> test: `1m29s`, train: `2m55s`, total: `4m24s`|1.0000 (0.00)|0.8333 (0.00)|`no data`|
| `BERT + DIET(seq) + ResponseSelector(t2t)`<br> test: `2m29s`, train: `3m55s`, total: `5m24s`|1.0000 (0.00)|0.8333 (0.00)|`no data`|


"""  # noqa E501
    assert output == expected_output


def test_comment_core():
    cmd = (
        "gomplate "
        f"-d data={TEST_DATA_DIR}/report_listformat_core.json "
        f"-d results_main={TEST_DATA_DIR}/report-on-schedule-2022-02-02.json "
        f"-f {TEMPLATE_FPATH}"
    )
    output = subprocess.check_output(cmd.split(" "), cwd=REPO_DIR)
    output = output.decode("utf-8")
    expected_output = """
Dataset: `RasaHQ/retail-demo`, Dataset repository branch: `fix-model-regression-tests` (external repository), commit: `8226b51b4312aa4d3723098cf6d4028feea040b4`
Configuration repository branch: `main`

| Dialog Policy Configuration | Action Level Micro Avg. F1 | Conversation Level Accuracy | Run Time Train | Run Time Test |
|---------------|-----------------|-----------------|-------------------|-------------------|
| `Rules + Memo + TED` |1.0000 (0.00)|1.0000 (0.00)|`4m27s`| `31s`|
| `Rules + Memo + TED` |1.0000 (0.00)|1.0000 (0.00)|`5m27s`| `41s`|

"""  # noqa E501
    assert output == expected_output
