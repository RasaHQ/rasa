from typing import Any, Dict, Text

from sanic.app import Sanic

# mypy check fails here but it actually successfully loads the initial module
# so it's probably an internal issue of mypy with no repercussions
from sanic.blueprints import Blueprint as SanicBlueprint  # type: ignore[attr-defined]

class Blueprint(SanicBlueprint):
    def register(self, app: Sanic, options: Dict[Text, Any]) -> None: ...
