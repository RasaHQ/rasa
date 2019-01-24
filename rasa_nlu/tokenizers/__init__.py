class Tokenizer(object):
    pass


class Token(object):
    def __init__(self, text, offset, data=None):
        self.offset = offset
        self.text = text
        self.end = offset + len(text)
        self.data = data if data else {}

    def set(self, prop, info):
        self.data[prop] = info

    def get(self, prop, default=None):
        return self.data.get(prop, default)
