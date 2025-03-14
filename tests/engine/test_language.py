import pytest

from rasa.engine.language import CUSTOM_LANGUAGE_CODE_PREFIX, Language
from rasa.shared.exceptions import RasaException

ENGLISH_LANGUAGE_CODE = "en"
US_ENGLISH_LANGUAGE_CODE = "en-US"
ENGLISH_LANGUAGE_LABEL = "English"
US_ENGLISH_LANGUAGE_LABEL = "English (United States)"


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

    assert "not a BCP 47 standard language code" in str(exc_info.value)


def test_valid_custom_language() -> None:
    """Test that a custom language code is correctly parsed."""
    custom_language_code = f"x-{ENGLISH_LANGUAGE_CODE}-formal"
    lang = Language.from_language_code(custom_language_code)
    assert lang.code == custom_language_code
    assert lang.label == ENGLISH_LANGUAGE_LABEL


def test_custom_language_with_hyphen_language_code() -> None:
    """Test that a custom language code is correctly parsed."""
    custom_language_code = f"x-{US_ENGLISH_LANGUAGE_CODE}-formal"
    lang = Language.from_language_code(custom_language_code)
    assert lang.code == custom_language_code
    assert lang.label == US_ENGLISH_LANGUAGE_LABEL


def test_custom_language_with_underscore_language_code() -> None:
    """Test that a custom language code is correctly parsed."""
    custom_language_code = "x-en_US-formal"
    with pytest.raises(RasaException) as exc_info:
        Language.from_language_code(custom_language_code)

    expected = (
        "Base language 'en_US' in custom language "
        "'x-en_US-formal' is not a valid language code."
    )
    assert expected == str(exc_info.value)


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
        Language.validate_custom_language_code(invalid_custom_language_code)

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


def test_is_language_code_bcp_47_standard_valid() -> None:
    """Test that valid BCP 47 standard language codes are correctly detected."""
    valid_codes = ["en", "de", "fr", "en-US", "zh-Hant", "es-419"]
    for code in valid_codes:
        assert Language.is_language_code_bcp_47_standard(code)


def test_is_language_code_bcp_47_standard_invalid() -> None:
    """Test that non-standard language codes are correctly rejected."""
    invalid_codes = ["EN", "english", "eng-USA", "deutsch", "en_US"]
    for code in invalid_codes:
        assert not Language.is_language_code_bcp_47_standard(code)
