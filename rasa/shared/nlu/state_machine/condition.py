from rasa_sdk import Tracker
import abc


class Condition(abc.ABC):
    @abc.abstractmethod
    def is_valid(self, tracker: Tracker):
        raise NotImplementedError()