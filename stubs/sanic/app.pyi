# mypy check fails here but it actually successfully loads the initial module
# so it's probably an internal issue of mypy with no repercussions
from sanic.app import Sanic as SanicSanic  # type: ignore[attr-defined]

class Sanic(SanicSanic):
    def stop(self) -> None: ...
