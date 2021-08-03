from __future__ import annotations

from typing import Dict, Optional, Text, Any, List

import rasa.shared.utils.io
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource


class AddInputs(GraphComponent):
    default_config = {}

    @classmethod
    def create(
        cls,
        config: Dict,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> AddInputs:
        return cls()

    def add(self, i1: int, i2: int) -> int:
        return i1 + i2


class SubtractByX(GraphComponent):
    default_config = {"x": 0}

    def __init__(self, x: int) -> None:
        self._x = x

    @classmethod
    def create(
        cls,
        config: Dict,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> SubtractByX:
        return cls(config["x"])

    def subtract_x(self, i: int) -> int:
        return i - self._x


class ProvideX(GraphComponent):
    default_config = {}

    def __init__(self) -> None:
        self.x = 1

    @classmethod
    def create(
        cls,
        config: Dict,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        x: Optional[int] = None,
        **kwargs: Any,
    ) -> ProvideX:
        instance = cls()
        if x:
            instance.x = x
        return instance

    @classmethod
    def create_with_2(
        cls,
        config: Dict,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> ProvideX:
        return cls.create(
            config, model_storage, resource, execution_context, 2, **kwargs
        )

    def provide(self) -> int:
        return self.x


class ExecutionContextAware(GraphComponent):
    default_config = {}

    def __init__(self, model_id: Text) -> None:
        self.model_id = model_id

    @classmethod
    def create(
        cls,
        config: Dict,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> ExecutionContextAware:
        return cls(execution_context.model_id)

    def get_model_id(self) -> Text:
        return self.model_id


class PersistableTestComponent(GraphComponent):
    default_config = {}

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        eager_instantiated_value: Any = None,
    ) -> None:
        self._model_storage = model_storage
        self._resource = resource
        self._config = config
        self._eager_instantiated_value = eager_instantiated_value

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> PersistableTestComponent:
        assert model_storage
        assert resource

        return cls(config, model_storage, resource)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> PersistableTestComponent:
        assert model_storage
        assert resource

        with model_storage.read_from(resource) as directory:
            eager_instantiated_value = rasa.shared.utils.io.read_json_file(
                directory / "test.json"
            )
        return cls(config, model_storage, resource, eager_instantiated_value)

    def supported_languages(self) -> List[Text]:
        return []

    def required_packages(self) -> List[Text]:
        return []

    def train(self) -> Resource:
        with self._model_storage.write_to(self._resource) as directory:
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                directory / "test.json", self._config["test_value"]
            )
            sub_dir = directory / "sub_dir"
            sub_dir.mkdir()

            rasa.shared.utils.io.dump_obj_as_json_to_file(
                sub_dir / "test.json", self._config["test_value_for_sub_directory"]
            )

        return self._resource

    def run_train_process(self) -> Any:
        return self._eager_instantiated_value

    def run_inference(self) -> Any:
        return self._eager_instantiated_value
