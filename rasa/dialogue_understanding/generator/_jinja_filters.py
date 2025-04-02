import json


def to_json_escaped_string(s: str) -> str:
    """
    Serializes the input string using JSON encoding to escape special characters
    like newlines (\\n), tabs (\\t), and quotes.
    """
    return json.dumps(s, ensure_ascii=False)
