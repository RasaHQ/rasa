import os
import pytest

import rasa.utils.io
from rasa.core import utils


@pytest.fixture(scope="session")
def loop():
    from pytest_sanic.plugin import loop as sanic_loop

    return rasa.utils.io.enable_async_loop_debugging(next(sanic_loop()))


def test_is_int():
    assert utils.is_int(1)
    assert utils.is_int(1.0)
    assert not utils.is_int(None)
    assert not utils.is_int(1.2)
    assert not utils.is_int("test")


def test_subsample_array_read_only():
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    r = utils.subsample_array(t, 5, can_modify_incoming_array=False)

    assert len(r) == 5
    assert set(r).issubset(t)


def test_subsample_array():
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # this will modify the original array and shuffle it
    r = utils.subsample_array(t, 5)

    assert len(r) == 5
    assert set(r).issubset(t)


def test_on_hot():
    r = utils.one_hot(4, 6)
    assert (r[[0, 1, 2, 3, 5]] == 0).all()
    assert r[4] == 1


def test_on_hot_out_of_range():
    with pytest.raises(ValueError):
        utils.one_hot(4, 3)


def test_cap_length():
    assert utils.cap_length("mystring", 6) == "mys..."


def test_cap_length_without_ellipsis():
    assert utils.cap_length("mystring", 3, append_ellipsis=False) == "mys"


def test_cap_length_with_short_string():
    assert utils.cap_length("my", 3) == "my"


def test_pad_list_to_size():
    assert utils.pad_list_to_size(["e1", "e2"], 4, "other") == [
        "e1",
        "e2",
        "other",
        "other",
    ]


def test_read_lines():
    lines = utils.read_lines(
        "data/test_stories/stories.md", max_line_limit=2, line_pattern=r"\*.*"
    )

    lines = list(lines)

    assert len(lines) == 2


os.environ["USER_NAME"] = "user"
os.environ["PASS"] = "pass"


def test_read_yaml_string():
    config_without_env_var = """
    user: user
    password: pass
    """
    r = rasa.utils.io.read_yaml(config_without_env_var)
    assert r["user"] == "user" and r["password"] == "pass"


def test_read_yaml_string_with_env_var():
    config_with_env_var = """
    user: ${USER_NAME}
    password: ${PASS}
    """
    r = rasa.utils.io.read_yaml(config_with_env_var)
    assert r["user"] == "user" and r["password"] == "pass"


def test_read_yaml_string_with_multiple_env_vars_per_line():
    config_with_env_var = """
    user: ${USER_NAME} ${PASS}
    password: ${PASS}
    """
    r = rasa.utils.io.read_yaml(config_with_env_var)
    assert r["user"] == "user pass" and r["password"] == "pass"


def test_read_yaml_string_with_env_var_prefix():
    config_with_env_var_prefix = """
    user: db_${USER_NAME}
    password: db_${PASS}
    """
    r = rasa.utils.io.read_yaml(config_with_env_var_prefix)
    assert r["user"] == "db_user" and r["password"] == "db_pass"


def test_read_yaml_string_with_env_var_postfix():
    config_with_env_var_postfix = """
    user: ${USER_NAME}_admin
    password: ${PASS}_admin
    """
    r = rasa.utils.io.read_yaml(config_with_env_var_postfix)
    assert r["user"] == "user_admin" and r["password"] == "pass_admin"


def test_read_yaml_string_with_env_var_infix():
    config_with_env_var_infix = """
    user: db_${USER_NAME}_admin
    password: db_${PASS}_admin
    """
    r = rasa.utils.io.read_yaml(config_with_env_var_infix)
    assert r["user"] == "db_user_admin" and r["password"] == "db_pass_admin"


def test_read_yaml_string_with_env_var_not_exist():
    config_with_env_var_not_exist = """
    user: ${USER_NAME}
    password: ${PASSWORD}
    """
    with pytest.raises(ValueError):
        rasa.utils.io.read_yaml(config_with_env_var_not_exist)


@pytest.mark.parametrize("file, parents", [("A/test.md", "A"), ("A", "A")])
def test_file_in_path(file, parents):
    assert rasa.utils.io.is_subdirectory(file, parents)


@pytest.mark.parametrize(
    "file, parents", [("A", "A/B"), ("B", "A"), ("A/test.md", "A/B"), (None, "A")]
)
def test_file_not_in_path(file, parents):
    assert not rasa.utils.io.is_subdirectory(file, parents)
