import glob
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from pytest import FixtureRequest, MonkeyPatch


def input_data_file(request: FixtureRequest, name):
    test_data_dir = Path(request.config.rootpath, "data", "test_data_convert_e2e")
    file = (test_data_dir / name).absolute()
    return file


def output_files(request: FixtureRequest):
    output_file_path = Path(request.config.rootpath, "e2e_tests")
    extension = ".yml"
    files = glob.glob(str(output_file_path) + "/" f"*{extension}")
    return files


def run_command(request: FixtureRequest, name, sheet_name):
    data_file = input_data_file(request, name)
    command = subprocess.Popen(
        "rasa data convert e2e " + str(data_file) + " --sheet-name=" + sheet_name,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return command.communicate()


def read_csv_file(request: FixtureRequest, name):
    data_file = input_data_file(request, name)
    doc = pd.read_csv(data_file).to_dict()
    return doc


def read_yml_files(request: FixtureRequest):
    file_names = output_files(request)
    with open(file_names[0], "r") as file:
        yml_dict = yaml.safe_load(file)
    return yml_dict


def read_xlsx_file(request: FixtureRequest, name):
    data_file = input_data_file(request, name)
    doc = pd.read_excel(data_file).to_dict()
    return doc


def test_rasa_data_convert_e2e_feature_enabled_csv(
    request: FixtureRequest,
    monkeypatch: MonkeyPatch,
):
    """
    Test captures the subprocess output for the command run.
    validates user message is displayed when `RASA_PRO_BETA_E2E_CONVERSION=true`
    validates yml file is created when cli command is successful run
    with sample .csv file
    """
    csv_file_name = "sample_conversations.csv"
    monkeypatch.setenv("RASA_PRO_BETA_E2E_CONVERSION", "true")

    result = run_command(request, name=csv_file_name, sheet_name="")
    assert (
        "\\n\\nIf you want to disable this beta feature, "
        "set the environment variable\\n`RASA_PRO_BETA_E2E_CONVERSION=false`.\\n\\n"
    ) in str(result[0])
    assert "output_file" in str(result[0])

    file_names = output_files(request)
    assert bool(file_names) is True
    input_dict = read_csv_file(request, name=csv_file_name)
    output_dict = read_yml_files(request)

    input_utterance_lines = input_dict["conversation"][0]
    if input_utterance_lines.startswith("User"):
        line_text_user = input_utterance_lines.split(":")[1].strip("\nFinbo").strip(" ")
        output_utterance_lines_user = output_dict.get("test_cases", [])[0]["steps"][0][
            "user"
        ]
        assert line_text_user == output_utterance_lines_user
    if input_utterance_lines.find("Finbo"):
        line_text_bot = input_utterance_lines.split(":")[2].strip("\nFinbo").strip(" ")
        output_utterance_lines_bot = output_dict.get("test_cases", [])[0]["steps"][0][
            "assertions"
        ][0]["bot_uttered"]["text_matches"]
        assert line_text_bot == output_utterance_lines_bot


def test_rasa_data_convert_e2e_feature_enabled_xlsx(
    request: FixtureRequest,
    monkeypatch: MonkeyPatch,
):
    """
    Test captures the subprocess output for the command run.
    validates user message is displayed when `RASA_PRO_BETA_E2E_CONVERSION=true`
    validates yml file is created when cli command is successful run
    with sample .xlsx file
    """
    xlsx_file_name = "sample_conversations.xlsx"
    monkeypatch.setenv("RASA_PRO_BETA_E2E_CONVERSION", "true")
    result = run_command(request, name=xlsx_file_name, sheet_name="Sheet1")
    assert (
        "\\n\\nIf you want to disable this beta feature, "
        "set the environment variable\\n`"
        "RASA_PRO_BETA_E2E_CONVERSION=false`.\\n\\n"
    ) in str(result[0])
    assert "output_file" in str(result[0])
    file_names = output_files(request)
    assert bool(file_names) is True
    input_dict = read_xlsx_file(request, name=xlsx_file_name)
    output_dict = read_yml_files(request)
    input_utterance_lines = input_dict["conversation"][0]
    if input_utterance_lines.startswith("User"):
        line_text_user = input_utterance_lines.split(":")[1].strip("\nFinbo").strip(" ")
        output_utterance_lines_user = output_dict.get("test_cases", [])[0]["steps"][0][
            "user"
        ]
        assert line_text_user == output_utterance_lines_user
    if input_utterance_lines.find("Finbo"):
        line_text_bot = input_utterance_lines.split(":")[2].strip("\nFinbo").strip(" ")
        output_utterance_lines_bot = output_dict.get("test_cases", [])[0]["steps"][0][
            "assertions"
        ][0]["bot_uttered"]["text_matches"]
        assert line_text_bot == output_utterance_lines_bot


def test_rasa_data_convert_e2e_feature_disabled(
    request: FixtureRequest,
    monkeypatch: MonkeyPatch,
):
    """
    Test captures the subprocess output for the command run.
    validates user message is displayed when `RASA_PRO_BETA_E2E_CONVERSION=false`
    """
    monkeypatch.setenv("RASA_PRO_BETA_E2E_CONVERSION", "false")
    result = run_command(request, name="test_sample_conversations.csv", sheet_name="")
    assert (
        "\\n\\nYou need to explicitly enable the conversion of sample conversations"
        " into end-to-end tests feature, "
        "before\\nusage. Set the `RASA_PRO_BETA_E2E_CONVERSION=true` "
        "environment variable before\\nrunning the command again.\\n"
    ) in str(result[0])
