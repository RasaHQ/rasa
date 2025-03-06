import pytest

from rasa.engine.language import CUSTOM_LANGUAGE_CODE_PREFIX, Language
from rasa.shared.exceptions import RasaException

ENGLISH_LANGUAGE_CODE = "en"
ENGLISH_LANGUAGE_LABEL = "English"


def test_valid_standard_language() -> None:
    """Test that a standard language code is correctly parsed."""
    lang = Language.from_language_code(ENGLISH_LANGUAGE_CODE)
    assert lang.code == ENGLISH_LANGUAGE_CODE
    assert lang.label == ENGLISH_LANGUAGE_LABEL
    assert lang.is_default is False


def test_valid_standard_language_default() -> None:
    """Test that a standard language code can be marked as default."""
    lang = Language.from_language_code(ENGLISH_LANGUAGE_CODE, is_default=True)
    assert lang.is_default is True


def test_invalid_standard_language_code() -> None:
    """Test that an invalid language code raises an exception."""
    with pytest.raises(RasaException) as exc_info:
        Language.from_language_code(ENGLISH_LANGUAGE_LABEL)

    assert "not a valid language code" in str(exc_info.value)


def test_valid_custom_language() -> None:
    """Test that a custom language code is correctly parsed."""
    custom_language_code = f"x-{ENGLISH_LANGUAGE_CODE}-formal"
    lang = Language.from_language_code(custom_language_code)
    assert lang.code == custom_language_code
    assert lang.label == ENGLISH_LANGUAGE_LABEL


def test_invalid_custom_language_format() -> None:
    """Test that an invalid custom language code raises an exception."""
    invalid_custom_language_code = "x-en"
    with pytest.raises(RasaException) as exc_info:
        Language.from_language_code(invalid_custom_language_code)

    expected = (
        f"must be in the format "
        f"'{CUSTOM_LANGUAGE_CODE_PREFIX}<language_code>-<custom_label>"
    )
    assert expected in str(exc_info.value)


def test_invalid_custom_language_empty_label() -> None:
    """Test that an empty custom label raises an exception."""
    invalid_custom_language_code = "x-en-"
    with pytest.raises(RasaException) as exc_info:
        Language.from_language_code(invalid_custom_language_code)

    assert "cannot be empty." in str(exc_info.value)


def test_invalid_custom_language_prefix() -> None:
    """Test that a custom language code with an invalid prefix raises an exception."""
    invalid_custom_language_code = "y-en-formal"
    with pytest.raises(RasaException) as exc_info:
        Language.validate_custom_language(invalid_custom_language_code)

    assert f"must start with '{CUSTOM_LANGUAGE_CODE_PREFIX}" in str(exc_info.value)


def test_invalid_custom_language_base() -> None:
    """Test that an invalid base language code raises an exception."""
    invalid_base_language = "foo"
    invalid_custom_language_code = f"x-{invalid_base_language}-formal"
    with pytest.raises(RasaException) as exc_info:
        Language.from_language_code(invalid_custom_language_code)

    assert "is not a valid language code" in str(exc_info.value)


def test_as_dict() -> None:
    """Test that a language object can be converted to a dictionary."""
    lang = Language.from_language_code(ENGLISH_LANGUAGE_CODE)
    expected = {
        "code": ENGLISH_LANGUAGE_CODE,
        "label": ENGLISH_LANGUAGE_LABEL,
        "is_default": False,
    }
    assert lang.as_dict() == expected
