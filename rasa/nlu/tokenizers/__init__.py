import functools


class Tokenizer:
    pass


@functools.total_ordering
class Token:
    def __init__(self, text, offset, data=None):
        self.offset = offset
        self.text = text
        self.end = offset + len(text)
        self.data = data if data else {}

    def set(self, prop, info):
        self.data[prop] = info

    def get(self, prop, default=None):
        return self.data.get(prop, default)

    def __eq__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return (self.offset, self.end, self.text) == (
            other.offset,
            other.end,
            other.text,
        )

    def __lt__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return (self.offset, self.end, self.text) < (
            other.offset,
            other.end,
            other.text,
        )
