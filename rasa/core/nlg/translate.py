from typing import Any, Dict, List, Optional, Text

from rasa.engine.language import Language
from rasa.shared.core.flows.constants import KEY_TRANSLATION


def get_translated_text(
    text: Optional[Text],
    translation: Dict[Text, Any],
    language: Optional[Language] = None,
) -> Optional[Text]:
    """Get the translated text from the message.

    Args:
        text: The default text to use if no translation is found.
        translation: The translations for the text.
        language: The language to use for the translation.

    Returns:
        The translated text if found, otherwise the default text.
    """
    language_code = language.code if language else None
    return translation.get(language_code, text)


def has_translation(
    message: Dict[Text, Any], language: Optional[Language] = None
) -> bool:
    """Check if the message has a translation for the given language."""
    language_code = language.code if language else None
    return language_code in message.get(KEY_TRANSLATION, {})


def get_translated_buttons(
    buttons: Optional[List[Dict[Text, Any]]], language: Optional[Language] = None
) -> Optional[List[Dict[Text, Any]]]:
    """Get the translated buttons from the message.

    Args:
        buttons: The default buttons to use if no translation is found.
        language: The language to use for the translation.

    Returns:
        The translated buttons if found; otherwise, the default buttons.
    """
    if buttons is None:
        return None

    language_code = language.code if language else None
    translated_buttons = []
    for button in buttons:
        translation = button.get(KEY_TRANSLATION, {})
        language_translation = translation.get(language_code, {})

        # Maintain the original key order to ensure
        # accurate comparisons of BotUtter events.
        translated_button = {
            key: language_translation.get(key, button.get(key))
            for key, value in button.items()
            if key != KEY_TRANSLATION
        }
        translated_buttons.append(translated_button)
    return translated_buttons
