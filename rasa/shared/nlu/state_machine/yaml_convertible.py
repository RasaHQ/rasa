import abc


class YAMLConvertable(abc.ABC):
    @abc.abstractmethod
    def as_yaml(self):
        "Save to YAML"
        pass