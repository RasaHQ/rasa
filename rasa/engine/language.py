from dataclasses import dataclass
from typing import Any, Dict, Text

from langcodes import Language as LangcodesLanguage

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
        language = LangcodesLanguage.make(language_code)
        cls.validate_language(language)

        return cls(
            code=language_code,
            label=cls.get_language_label(language),
            is_default=is_default,
        )

    @staticmethod
    def get_language_label(language: LangcodesLanguage) -> str:
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

        if language_code.startswith(CUSTOM_LANGUAGE_CODE_PREFIX):
            # If it's a custom language, derive the label from the base language code.
            parts = language_code.split("-")
            base_language_code = parts[1]
            base_language = LangcodesLanguage.make(base_language_code)
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
            cls.validate_custom_language(language_code)

    @staticmethod
    def validate_custom_language(custom_language_code: str) -> None:
        """Validates a custom language code.

        A valid custom language code should adhere to the format:
          "x-<existing_language_code>-<custom_label>"
        Example: x-en-formal

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

        # Split the language code into parts.
        parts = custom_language_code.split("-")
        if len(parts) != 3:
            raise RasaException(
                f"Custom language '{custom_language_code}' must be in the format "
                f"'{CUSTOM_LANGUAGE_CODE_PREFIX}<language_code>-<custom_label>'."
            )

        # Validate the base language code using langcodes.
        base_language_code = parts[1]
        base_language = LangcodesLanguage.make(base_language_code)
        if not base_language.is_valid():
            raise RasaException(
                f"Base language '{base_language_code}' in custom language "
                f"'{custom_language_code}' is not a valid language code."
            )

        # Ensure the custom label is not empty.
        custom_label = parts[2]
        if not custom_label:
            raise RasaException(
                f"Custom label in custom language "
                f"'{custom_language_code}' cannot be empty."
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
