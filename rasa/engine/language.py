from dataclasses import dataclass
from typing import Any, Dict, Text

from langcodes import Language as LangcodesLanguage
from langcodes import standardize_tag
from langcodes.tag_parser import LanguageTagError

from rasa.shared.exceptions import RasaException

CUSTOM_LANGUAGE_CODE_PREFIX = "x-"


@dataclass(frozen=True)
class Language:
    code: str
    label: str
    is_default: bool

    @classmethod
    def from_language_code(
        cls, language_code: str, is_default: bool = False
    ) -> "Language":
        """Creates a Language object from a language code.

        Args:
            language_code: The language code.
            is_default: Whether the language is the default language.

        Returns:
            A Language object.

        Raises:
            RasaException: If the language code or custom language code is invalid.
        """
        if cls.is_custom_language_code(language_code):
            cls.validate_custom_language_code(language_code)
        elif not cls.is_language_code_bcp_47_standard(language_code):
            raise RasaException(
                f"Language '{language_code}' is not a BCP 47 standard language code."
            )

        language = LangcodesLanguage.get(language_code)
        return cls(
            code=language_code,
            label=cls.get_language_label(language),
            is_default=is_default,
        )

    @staticmethod
    def is_language_code_bcp_47_standard(language_code: str) -> bool:
        """Checks if a language code is a BCP 47 standard language code.

        Args:
            language_code: The language code to check.

        Returns:
            `True` if the language code is a BCP 47 standard, `False` otherwise.
        """
        try:
            standardized_language_code = standardize_tag(language_code)
            return (
                standardized_language_code == language_code
                and LangcodesLanguage.get(language_code).is_valid()
            )
        except LanguageTagError:
            return False

    @staticmethod
    def is_custom_language_code(language_code: str) -> bool:
        """Checks if a language code is a custom language code.

        Args:
            language_code: The language code to check.

        Returns:
            `True` if the language code is a custom language code, `False` otherwise.
        """
        return language_code.startswith(CUSTOM_LANGUAGE_CODE_PREFIX)

    @classmethod
    def get_language_label(cls, language: LangcodesLanguage) -> str:
        """Gets the display name of a language.

        For custom languages (in the format "x-<base_lang>-<custom_label>"),
        the label is derived from the base language code.
        This method considers that the language code has previously been validated.

        Args:
            language: The language code.

        Returns:
            The display name of the language.
        """
        language_code = str(language)

        if cls.is_custom_language_code(language_code):
            # If it's a custom language, derive the label from the base language code.
            without_prefix = language_code[len(CUSTOM_LANGUAGE_CODE_PREFIX) :]
            base_language_code, _ = without_prefix.rsplit("-", 1)
            base_language = LangcodesLanguage.get(base_language_code)
            return base_language.display_name()
        else:
            return language.display_name()

    @classmethod
    def validate_language(cls, language: LangcodesLanguage) -> None:
        """Validates a language code.

        Args:
            language: The language object to validate.

        Raises:
            RasaException: If the language validation fails.
        """
        if not language.is_valid():
            raise RasaException(f"Language '{language}' is not a valid language code.")

        language_code = str(language)
        if language_code.startswith(CUSTOM_LANGUAGE_CODE_PREFIX):
            cls.validate_custom_language_code(language_code)

    @classmethod
    def validate_custom_language_code(cls, custom_language_code: str) -> None:
        """Validates a custom language code.

        A valid custom language code should adhere to the format:
          "x-<existing_language_code>-<custom_label>"
        Example: x-en-formal or x-en-US-formal.

        Args:
            custom_language_code: The custom language code to validate.

        Raises:
            RasaException: If the custom language code validation fails.
        """
        # Ensure the custom language code starts with the custom prefix.
        if not custom_language_code.startswith(CUSTOM_LANGUAGE_CODE_PREFIX):
            raise RasaException(
                f"Custom language '{custom_language_code}' must "
                f"start with '{CUSTOM_LANGUAGE_CODE_PREFIX}'."
            )

        # Remove the custom prefix.
        without_prefix = custom_language_code[len(CUSTOM_LANGUAGE_CODE_PREFIX) :]
        if "-" not in without_prefix:
            raise RasaException(
                f"Custom language '{custom_language_code}' must be in the format "
                f"'{CUSTOM_LANGUAGE_CODE_PREFIX}<language_code>-<custom_label>'."
            )

        base_language_code, custom_label = without_prefix.rsplit("-", 1)
        if not base_language_code:
            raise RasaException(
                f"Base language in '{custom_language_code}' cannot be empty. "
                f"Expected custom language code format is "
                f"'{CUSTOM_LANGUAGE_CODE_PREFIX}<language_code>-<custom_label>'."
            )
        if not custom_label:
            raise RasaException(
                f"Custom label in '{custom_language_code}' cannot be empty."
                f"Expected custom language code format is "
                f"'{CUSTOM_LANGUAGE_CODE_PREFIX}<language_code>-<custom_label>'."
            )

        # Validate the base language code using langcodes.
        if not cls.is_language_code_bcp_47_standard(base_language_code):
            raise RasaException(
                f"Base language '{base_language_code}' in custom language "
                f"'{custom_language_code}' is not a valid language code."
            )

    def as_dict(self) -> Dict[Text, Any]:
        """Converts the Language dataclass instance into a dictionary.

        Returns:
            A dictionary representing the Language object.
        """
        return {
            "code": self.code,
            "label": self.label,
            "is_default": self.is_default,
        }
