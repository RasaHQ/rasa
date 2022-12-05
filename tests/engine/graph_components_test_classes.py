from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Text, Any, List

import rasa.shared.utils.io
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource


class AddInputs(GraphComponent):
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

    def add(self, i1: Any, i2: Any) -> int:
        return int(i1) + int(i2)


class SubtractByX(GraphComponent):
    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {"x": 0}

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

    def subtract_x(self, i: Any) -> int:
        return int(i) - self._x


class AssertComponent(GraphComponent):
    def __init__(self, value_to_assert: Any) -> None:
        self._value_to_assert = value_to_assert

    @classmethod
    def create(
        cls,
        config: Dict,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> AssertComponent:
        return cls(config["value_to_assert"])

    def run_assert(self, i: Any) -> CacheableText:
        assert i == self._value_to_assert
        return CacheableText("")


class ProvideX(GraphComponent):
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


class FileReader(GraphComponent):
    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> FileReader:
        return cls(Path(config["file_path"]))

    def read(self) -> CacheableText:
        return CacheableText(self._file_path.read_text())


class ExecutionContextAware(GraphComponent):
    def __init__(self, execution_context: ExecutionContext) -> None:
        self._execution_context = execution_context

    @classmethod
    def create(
        cls,
        config: Dict,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> ExecutionContextAware:
        return cls(execution_context)

    def get_execution_context(self) -> ExecutionContext:
        return self._execution_context


class PersistableTestComponent(GraphComponent):
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
        self._wrap_cacheable = self._config.get("wrap_output_in_cacheable", False)
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

    @staticmethod
    def required_packages() -> List[Text]:
        return []

    def train(self) -> Resource:
        with self._model_storage.write_to(self._resource) as directory:
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                directory / "test.json", self._config["test_value"]
            )
            sub_dir = directory / "sub_dir"
            sub_dir.mkdir()

            rasa.shared.utils.io.dump_obj_as_json_to_file(
                sub_dir / "test.json", self._config.get("test_value_for_sub_directory")
            )

        return self._resource

    def run_train_process(self) -> Any:
        if self._wrap_cacheable:
            return CacheableText(self._eager_instantiated_value)
        return self._eager_instantiated_value

    def run_inference(self) -> Any:
        if self._wrap_cacheable:
            return CacheableText(self._eager_instantiated_value)
        return self._eager_instantiated_value


class CacheableText:
    def __init__(self, text: Text) -> None:
        self.text = text

    def to_cache(self, directory: Path, model_storage: ModelStorage) -> None:
        rasa.shared.utils.io.write_text_file(self.text, directory / "my_file.txt")

    @classmethod
    def from_cache(
        cls,
        node_name: Text,
        directory: Path,
        model_storage: ModelStorage,
        output_fingerprint: Text,
    ) -> CacheableText:
        text = rasa.shared.utils.io.read_file(directory / "my_file.txt")
        return cls(text=text)

    def __repr__(self) -> Text:
        return self.text

    def __int__(self) -> int:
        return int(self.text)


class CacheableComponent(GraphComponent):
    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {"prefix": "Hello "}

    def __init__(self, prefix: Text):
        self.prefix = prefix

    @classmethod
    def create(
        cls,
        config: Dict,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> CacheableComponent:
        return cls(config["prefix"])

    def run(self, suffix: Text):
        return CacheableText(self.prefix + str(suffix))
