class Action:
    def __init__(self, name: str, send_domain: bool = True):
        self.__name = name
        self.__send_domain = send_domain

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value: str):
        self.__name = value

    @property
    def send_domain(self):
        return self.__send_domain

    @send_domain.setter
    def send_domain(self, value: bool):
        self.__send_domain = value
