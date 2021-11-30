from sanic.app import Sanic as SanicSanic


class Sanic(SanicSanic):

    def stop(self) -> None:
        ...
