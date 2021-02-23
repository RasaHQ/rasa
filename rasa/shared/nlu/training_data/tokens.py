from typing import Text, Optional, Dict, Any


class Token:
    """Represents a token in a text."""

    def __init__(
        self,
        text: Text,
        start: int,
        end: Optional[int] = None,
        data: Optional[Dict[Text, Any]] = None,
        lemma: Optional[Text] = None,
    ) -> None:
        """Initializes the token.

        Args:
            text: The text of the token.
            start: The start value of the token in the complete sentence.
            end: The end value of the token in the complete sentence.
            data: Optional data.
            lemma: The lemma of the token.
        """
        self.text = text
        self.start = start
        self.end = end if end else start + len(text)

        self.data = data if data else {}
        self.lemma = lemma or text

    def set(self, prop: Text, info: Any) -> None:
        """Sets a certain property inside data.

        Args:
            prop: The property to set.
            info: The value to set 'prop' to.
        """
        self.data[prop] = info

    def get(self, prop: Text, default: Optional[Any] = None) -> Any:
        """Gets a certain property from data.

        Args:
            prop: The property to return.
            default: The default value in case 'prop' is not present.

        Returns:
            The value of the property.
        """
        return self.data.get(prop, default)

    @classmethod
    def from_dict(cls, token_dict: Dict[Text, Any]) -> "Token":
        """Creates a token from the given dict.

        Args:
            token_dict: the dictionary that contains all properties of a token.

        Returns:
            A token.
        """
        return Token(
            token_dict["text"],
            token_dict["start"],
            token_dict["end"] if "end" in token_dict else None,
            token_dict["data"] if "data" in token_dict else None,
            token_dict["lemma"] if "lemma" in token_dict else None,
        )

    def __eq__(self, other: Any) -> bool:
        """Compares this token to another token."""
        if not isinstance(other, Token):
            return NotImplemented
        return (self.start, self.end, self.text, self.lemma) == (
            other.start,
            other.end,
            other.text,
            other.lemma,
        )

    def __lt__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return (self.start, self.end, self.text, self.lemma) < (
            other.start,
            other.end,
            other.text,
            other.lemma,
        )

    def __str__(self) -> Text:
        """Return the string representation of this Token."""
        return f"{self.text} ({self.start}-{self.end})"
