import os
from pathlib import Path
from typing import Callable, Text, List, Set

import pytest

import rasa.shared
import rasa.shared.utils.io
from rasa.shared.utils import io as io_utils
from rasa.utils import io as io_utils


def test_raise_user_warning():
    with pytest.warns(UserWarning) as record:
        io_utils.raise_warning("My warning.")

    assert len(record) == 1
    assert record[0].message.args[0] == "My warning."


def test_raise_future_warning():
    with pytest.warns(FutureWarning) as record:
        io_utils.raise_warning("My future warning.", FutureWarning)

    assert len(record) == 1
    assert record[0].message.args[0] == "My future warning."


def test_raise_deprecation():
    with pytest.warns(DeprecationWarning) as record:
        io_utils.raise_warning("My warning.", DeprecationWarning)

    assert len(record) == 1
    assert record[0].message.args[0] == "My warning."
    assert isinstance(record[0].message, DeprecationWarning)


def test_read_file_with_not_existing_path():
    with pytest.raises(ValueError):
        rasa.shared.utils.io.read_file("some path")


@pytest.mark.parametrize(
    "list_function, expected",
    [
        (
            io_utils.list_directory,
            {"subdirectory", "subdirectory/sub_file.txt", "file.txt"},
        ),
        (io_utils.list_files, {"subdirectory/sub_file.txt", "file.txt"}),
        (io_utils.list_subdirectories, {"subdirectory"}),
    ],
)
def test_list_directory(
    tmpdir: Path, list_function: Callable[[Text], List[Text]], expected: Set[Text]
):
    subdirectory = tmpdir / "subdirectory"
    subdirectory.mkdir()

    sub_sub_directory = subdirectory / "subdirectory"
    sub_sub_directory.mkdir()

    sub_sub_file = sub_sub_directory / "sub_file.txt"
    sub_sub_file.write_text("", encoding=io_utils.DEFAULT_ENCODING)

    file1 = subdirectory / "file.txt"
    file1.write_text("", encoding="utf-8")

    hidden_directory = subdirectory / ".hidden"
    hidden_directory.mkdir()

    hidden_file = subdirectory / ".test.text"
    hidden_file.write_text("", encoding="utf-8")

    expected = {str(subdirectory / entry) for entry in expected}

    assert set(list_function(str(subdirectory))) == expected


def test_read_yaml_string():
    config_without_env_var = """
    user: user
    password: pass
    """
    content = rasa.shared.utils.io.read_yaml(config_without_env_var)
    assert content["user"] == "user" and content["password"] == "pass"


def test_read_yaml_string_with_env_var():
    config_with_env_var = """
    user: ${USER_NAME}
    password: ${PASS}
    """
    content = rasa.shared.utils.io.read_yaml(config_with_env_var)
    assert content["user"] == "user" and content["password"] == "pass"


def test_read_yaml_string_with_multiple_env_vars_per_line():
    config_with_env_var = """
    user: ${USER_NAME} ${PASS}
    password: ${PASS}
    """
    content = rasa.shared.utils.io.read_yaml(config_with_env_var)
    assert content["user"] == "user pass" and content["password"] == "pass"


def test_read_yaml_string_with_env_var_prefix():
    config_with_env_var_prefix = """
    user: db_${USER_NAME}
    password: db_${PASS}
    """
    content = rasa.shared.utils.io.read_yaml(config_with_env_var_prefix)
    assert content["user"] == "db_user" and content["password"] == "db_pass"


def test_read_yaml_string_with_env_var_postfix():
    config_with_env_var_postfix = """
    user: ${USER_NAME}_admin
    password: ${PASS}_admin
    """
    content = rasa.shared.utils.io.read_yaml(config_with_env_var_postfix)
    assert content["user"] == "user_admin" and content["password"] == "pass_admin"


def test_read_yaml_string_with_env_var_infix():
    config_with_env_var_infix = """
    user: db_${USER_NAME}_admin
    password: db_${PASS}_admin
    """
    content = rasa.shared.utils.io.read_yaml(config_with_env_var_infix)
    assert content["user"] == "db_user_admin" and content["password"] == "db_pass_admin"


def test_read_yaml_string_with_env_var_not_exist():
    config_with_env_var_not_exist = """
    user: ${USER_NAME}
    password: ${PASSWORD}
    """
    with pytest.raises(ValueError):
        rasa.shared.utils.io.read_yaml(config_with_env_var_not_exist)


def test_environment_variable_not_existing():
    content = "model: \n  test: ${variable}"
    with pytest.raises(ValueError):
        rasa.shared.utils.io.read_yaml(content)


def test_environment_variable_dict_without_prefix_and_postfix():
    os.environ["variable"] = "test"
    content = "model: \n  test: ${variable}"

    content = rasa.shared.utils.io.read_yaml(content)

    assert content["model"]["test"] == "test"


def test_environment_variable_in_list():
    os.environ["variable"] = "test"
    content = "model: \n  - value\n  - ${variable}"

    content = rasa.shared.utils.io.read_yaml(content)

    assert content["model"][1] == "test"


def test_environment_variable_dict_with_prefix():
    os.environ["variable"] = "test"
    content = "model: \n  test: dir/${variable}"

    content = rasa.shared.utils.io.read_yaml(content)

    assert content["model"]["test"] == "dir/test"


def test_environment_variable_dict_with_postfix():
    os.environ["variable"] = "test"
    content = "model: \n  test: ${variable}/dir"

    content = rasa.shared.utils.io.read_yaml(content)

    assert content["model"]["test"] == "test/dir"


def test_environment_variable_dict_with_prefix_and_with_postfix():
    os.environ["variable"] = "test"
    content = "model: \n  test: dir/${variable}/dir"

    content = rasa.shared.utils.io.read_yaml(content)

    assert content["model"]["test"] == "dir/test/dir"


def test_emojis_in_yaml():
    test_data = """
    data:
        - one 😁💯 👩🏿‍💻👨🏿‍💻
        - two £ (?u)\\b\\w+\\b f\u00fcr
    """
    content = rasa.shared.utils.io.read_yaml(test_data)

    assert content["data"][0] == "one 😁💯 👩🏿‍💻👨🏿‍💻"
    assert content["data"][1] == "two £ (?u)\\b\\w+\\b für"


def test_emojis_in_tmp_file():
    test_data = """
        data:
            - one 😁💯 👩🏿‍💻👨🏿‍💻
            - two £ (?u)\\b\\w+\\b f\u00fcr
        """
    test_file = io_utils.create_temporary_file(test_data)
    content = rasa.shared.utils.io.read_yaml_file(test_file)

    assert content["data"][0] == "one 😁💯 👩🏿‍💻👨🏿‍💻"
    assert content["data"][1] == "two £ (?u)\\b\\w+\\b für"


def test_read_emojis_from_json():
    import json

    d = {"text": "hey 😁💯 👩🏿‍💻👨🏿‍💻🧜‍♂️(?u)\\b\\w+\\b} f\u00fcr"}
    json_string = json.dumps(d, indent=2)

    content = rasa.shared.utils.io.read_yaml(json_string)

    expected = "hey 😁💯 👩🏿‍💻👨🏿‍💻🧜‍♂️(?u)\\b\\w+\\b} für"
    assert content.get("text") == expected


def test_bool_str():
    test_data = """
    one: "yes"
    two: "true"
    three: "True"
    """

    content = rasa.shared.utils.io.read_yaml(test_data)

    assert content["one"] == "yes"
    assert content["two"] == "true"
    assert content["three"] == "True"
