from pkg_resources import get_distribution

__version__ = get_distribution('rasa_nlu').version


class Interpreter(object):
    def parse(self, text):
        raise NotImplementedError()
