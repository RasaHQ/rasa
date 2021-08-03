from __future__ import annotations

from typing import Dict, Optional, Text

from rasa.engine.graph import ExecutionContext, GraphComponent


class AddInputs(GraphComponent):
    default_config = {}

    @classmethod
    def create(cls, config: Dict, execution_context: ExecutionContext) -> AddInputs:
        return cls()

    def add(self, i1: int, i2: int) -> int:
        return i1 + i2


class SubtractByX(GraphComponent):
    default_config = {"x": 0}

    def __init__(self, x: int) -> None:
        self._x = x

    @classmethod
    def create(cls, config: Dict, execution_context: ExecutionContext) -> SubtractByX:
        return cls(config["x"])

    def subtract_x(self, i: int) -> int:
        return i - self._x


class ProvideX(GraphComponent):
    default_config = {}

    def __init__(self) -> None:
        self.x = 1

    @classmethod
    def create(
        cls, config: Dict, execution_context: ExecutionContext, x: Optional[int] = None
    ) -> ProvideX:
        instance = cls()
        if x:
            instance.x = x
        return instance

    @classmethod
    def create_with_2(
        cls, config: Dict, execution_context: ExecutionContext
    ) -> ProvideX:
        return cls.create(config, execution_context, 2)

    def provide(self) -> int:
        return self.x


class ExecutionContextAware(GraphComponent):
    default_config = {}

    def __init__(self, model_id: Text) -> None:
        self.model_id = model_id

    @classmethod
    def create(
        cls, config: Dict, execution_context: ExecutionContext
    ) -> ExecutionContextAware:
        return cls(execution_context.model_id)

    def get_model_id(self) -> Text:
        return self.model_id
