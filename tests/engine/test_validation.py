from typing import Any, Dict, Text, Type, Optional, List

import pytest

from rasa.engine import validation
from rasa.engine.exceptions import GraphSchemaValidationException
from rasa.engine.graph import GraphComponent, ExecutionContext, GraphSchema, SchemaNode
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.training_data.training_data import TrainingData


class TestComponentWithoutRun(GraphComponent):
    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls()


class TestComponentWithRun(TestComponentWithoutRun):
    def run(self) -> TrainingData:
        pass


def create_test_schema(
    uses: Type,  # The unspecified type is on purpose to enable testing of invalid cases
    constructor_name: Text = "create",
    run_fn: Text = "run",
    needs: Optional[Dict[Text, Text]] = None,
    eager: bool = True,
    parent: Optional[Type[GraphComponent]] = None,
) -> GraphSchema:
    parent_node = {}
    if parent:
        parent_node = {
            "parent": SchemaNode(
                needs={}, uses=parent, constructor_name="create", fn="run", config={}
            )
        }
    # noinspection PyTypeChecker
    return GraphSchema(
        {
            "my_node": SchemaNode(
                needs=needs or {},
                uses=uses,
                eager=eager,
                constructor_name=constructor_name,
                fn=run_fn,
                config={},
            ),
            **parent_node,
        }
    )


def test_graph_component_is_no_graph_component():
    class MyComponent:
        def other(self) -> TrainingData:
            pass

    schema = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="implement .+ interface"):
        validation.validate(schema, language=None, is_train_graph=True)


def test_graph_component_fn_does_not_exist():
    schema = create_test_schema(uses=TestComponentWithRun, run_fn="some_fn")

    with pytest.raises(
        GraphSchemaValidationException, match="specified method 'some_fn'"
    ):
        validation.validate(schema, language=None, is_train_graph=True)


def test_graph_output_is_not_fingerprintable_int():
    class MyComponent(TestComponentWithoutRun):
        def run(self) -> int:
            pass

    schema = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="fingerprintable"):
        validation.validate(schema, language=None, is_train_graph=True)


def test_predict_graph_output_is_not_fingerprintable():
    class MyComponent(TestComponentWithoutRun):
        def run(self) -> int:
            pass

    schema = create_test_schema(uses=MyComponent)

    validation.validate(schema, language=None, is_train_graph=False)


def test_graph_output_is_not_fingerprintable_any():
    class MyComponent(TestComponentWithoutRun):
        def run(self) -> Any:
            pass

    schema = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="fingerprintable"):
        validation.validate(schema, language=None, is_train_graph=True)


def test_graph_output_is_not_fingerprintable_None():
    class MyComponent(TestComponentWithoutRun):
        def run(self) -> None:
            pass

    schema = create_test_schema(uses=MyComponent,)

    with pytest.raises(GraphSchemaValidationException, match="fingerprintable"):
        validation.validate(schema, language=None, is_train_graph=True)


def test_graph_with_forward_referenced_output_type():
    class MyComponent(TestComponentWithoutRun):
        # The non imported type annotation is on purpose so we can provoke a error in
        # the test
        def run(self) -> "UserUttered":  # noqa: F821
            pass

    schema = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="forward reference"):
        validation.validate(schema, language=None, is_train_graph=True)


def test_graph_output_missing_type_annotation():
    class MyComponent(TestComponentWithoutRun):
        def run(self):
            pass

    schema = create_test_schema(uses=MyComponent)

    with pytest.raises(
        GraphSchemaValidationException, match="does not have a type annotation"
    ):
        validation.validate(schema, language=None, is_train_graph=True)


def test_graph_with_fingerprintable_output():
    class MyComponent(TestComponentWithoutRun):
        def run(self) -> TrainingData:
            pass

    schema = create_test_schema(uses=MyComponent)

    validation.validate(schema, language=None, is_train_graph=True)


class MyTrainingData(TrainingData):
    pass


def test_graph_with_fingerprintable_output_subclass():
    class MyComponent(TestComponentWithoutRun):
        def run(self) -> MyTrainingData:
            pass

    schema = create_test_schema(uses=MyComponent)

    validation.validate(schema, language=None, is_train_graph=True)


def test_graph_constructor_missing():
    class MyComponent(TestComponentWithoutRun):
        def run(self) -> TrainingData:
            pass

    schema = create_test_schema(uses=MyComponent, constructor_name="invalid")

    with pytest.raises(
        GraphSchemaValidationException, match="specified method 'invalid'"
    ):
        validation.validate(schema, language=None, is_train_graph=True)


def test_graph_constructor_config_wrong_type():
    class MyComponent(TestComponentWithRun):
        @classmethod
        def create(
            cls,
            config: Dict[int, int],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
        ) -> GraphComponent:
            pass

    schema = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="incompatible type"):
        validation.validate(schema, language=None, is_train_graph=True)


def test_graph_constructor_resource_wrong_type():
    class MyComponent(TestComponentWithRun):
        @classmethod
        def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Dict,
            execution_context: ExecutionContext,
        ) -> GraphComponent:
            pass

    schema = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="incompatible type"):
        validation.validate(schema, language=None, is_train_graph=True)


def test_graph_constructor_model_storage_wrong_type():
    class MyComponent(TestComponentWithRun):
        @classmethod
        def create(
            cls,
            config: Dict[Text, Any],
            model_storage: Any,
            resource: Resource,
            execution_context: ExecutionContext,
        ) -> GraphComponent:
            pass

    schema = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="incompatible type"):
        validation.validate(schema, language=None, is_train_graph=True)


def test_graph_constructor_execution_context_wrong_type():
    class MyComponent(TestComponentWithRun):
        @classmethod
        def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: Any,
        ) -> GraphComponent:
            pass

    schema = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="incompatible type"):
        validation.validate(schema, language=None, is_train_graph=True)


@pytest.mark.parametrize(
    "current_language, supported_languages",
    [("de", ["en"]), ("en", ["zh", "fi"]), ("us", [])],
)
def test_graph_constructor_execution_not_supported_language(
    current_language: Text, supported_languages: Optional[List[Text]]
):
    class MyComponent(TestComponentWithRun):
        @staticmethod
        def supported_languages() -> Optional[List[Text]]:
            return supported_languages

    schema = create_test_schema(uses=MyComponent)

    with pytest.raises(
        GraphSchemaValidationException, match="does not support .* language"
    ):
        validation.validate(schema, language=current_language, is_train_graph=False)


@pytest.mark.parametrize(
    "current_language, supported_languages",
    [(None, None), ("en", ["zh", "en"]), ("zh", None), (None, ["en"])],
)
def test_graph_constructor_execution_supported_language(
    current_language: Optional[Text], supported_languages: Optional[List[Text]]
):
    class MyComponent(TestComponentWithRun):
        @staticmethod
        def supported_languages() -> Optional[List[Text]]:
            return supported_languages

    schema = create_test_schema(uses=MyComponent)

    validation.validate(schema, language=current_language, is_train_graph=False)


@pytest.mark.parametrize(
    "required_packages", [["pytorch"], ["tensorflow", "kubernetes"]]
)
def test_graph_missing_package_requirements(required_packages: List[Text]):
    class MyComponent(TestComponentWithRun):
        @staticmethod
        def required_packages() -> List[Text]:
            """Any extra python dependencies required for this component to run."""
            return required_packages

    schema = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="not installed"):
        validation.validate(schema, language=None, is_train_graph=True)


@pytest.mark.parametrize("required_packages", [["tensorflow"], ["tensorflow", "numpy"]])
def test_graph_satisfied_package_requirements(required_packages: List[Text]):
    class MyComponent(TestComponentWithRun):
        @staticmethod
        def required_packages() -> List[Text]:
            """Any extra python dependencies required for this component to run."""
            return required_packages

    schema = create_test_schema(uses=MyComponent)

    validation.validate(schema, language=None, is_train_graph=True)


def test_run_param_not_satisfied():
    class MyComponent(TestComponentWithoutRun):
        def run(self, some_param: TrainingData) -> TrainingData:
            pass

    schema = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="needs the param"):
        validation.validate(schema, language=None, is_train_graph=True)


def test_run_param_satifisfied_due_to_default():
    class MyComponent(TestComponentWithoutRun):
        def run(self, some_param: TrainingData = TrainingData()) -> TrainingData:
            pass

    schema = create_test_schema(uses=MyComponent)

    validation.validate(schema, language=None, is_train_graph=True)


def test_too_many_supplied_params():
    schema = create_test_schema(
        uses=TestComponentWithRun, needs={"some_param": "parent"}
    )

    with pytest.raises(
        GraphSchemaValidationException, match="does not accept a parameter"
    ):
        validation.validate(
            schema, language=None, is_train_graph=True,
        )


def test_too_many_supplied_params_but_kwargs():
    class MyComponent(TestComponentWithoutRun):
        def run(self, **kwargs: Any) -> TrainingData:
            pass

    schema = create_test_schema(
        uses=MyComponent, needs={"some_param": "parent"}, parent=TestComponentWithRun
    )

    validation.validate(schema, language=None, is_train_graph=True)


def test_run_fn_with_variable_length_positional_param():
    class MyComponent(TestComponentWithoutRun):
        def run(self, *args: Any, some_param: TrainingData) -> TrainingData:
            pass

    schema = create_test_schema(
        uses=MyComponent, needs={"some_param": "parent"}, parent=TestComponentWithRun
    )

    validation.validate(schema, language=None, is_train_graph=True)


def test_matching_params_due_to_constructor():
    class MyComponent(TestComponentWithRun):
        @classmethod
        def load(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
            some_param: TrainingData,
        ) -> GraphComponent:
            pass

    schema = create_test_schema(
        uses=MyComponent,
        needs={"some_param": "parent"},
        eager=False,
        constructor_name="load",
        parent=TestComponentWithRun,
    )

    validation.validate(
        schema, language=None, is_train_graph=True,
    )


def test_matching_params_due_to_constructor_but_eager():
    class MyComponent(TestComponentWithRun):
        @classmethod
        def load(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
            some_param: TrainingData,
        ) -> GraphComponent:
            pass

    schema = create_test_schema(
        uses=MyComponent,
        needs={"some_param": "parent"},
        eager=True,
        constructor_name="load",
    )

    with pytest.raises(GraphSchemaValidationException, match="lazy mode"):
        validation.validate(
            schema, language=None, is_train_graph=True,
        )


@pytest.mark.parametrize(
    "eager, error_message", [(True, "lazy mode"), (False, "needs the param")]
)
def test_unsatisfied_constructor(eager: bool, error_message: Text):
    class MyComponent(TestComponentWithRun):
        @classmethod
        def load(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
            some_param: TrainingData,
        ) -> GraphComponent:
            pass

    schema = create_test_schema(uses=MyComponent, eager=eager, constructor_name="load",)

    with pytest.raises(GraphSchemaValidationException, match=error_message):
        validation.validate(
            schema, language=None, is_train_graph=True,
        )


def test_parent_supplying_wrong_type():
    class MyUnreliableParent(TestComponentWithoutRun):
        def run(self) -> Domain:
            pass

    class MyComponent(TestComponentWithoutRun):
        def run(self, training_data: TrainingData) -> TrainingData:
            pass

    schema = create_test_schema(
        uses=MyComponent, parent=MyUnreliableParent, needs={"training_data": "parent"},
    )

    with pytest.raises(GraphSchemaValidationException, match="type .* expected"):
        validation.validate(schema, language=None, is_train_graph=True)


def test_parent_supplying_wrong_type_to_constructor():
    class MyUnreliableParent(TestComponentWithoutRun):
        def run(self) -> Domain:
            pass

    class MyComponent(TestComponentWithRun):
        @classmethod
        def load(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
            some_param: TrainingData,
        ) -> GraphComponent:
            pass

    schema = create_test_schema(
        uses=MyComponent,
        eager=False,
        constructor_name="load",
        parent=MyUnreliableParent,
        needs={"some_param": "parent"},
    )

    with pytest.raises(GraphSchemaValidationException, match="type .* expected"):
        validation.validate(schema, language=None, is_train_graph=True)


def test_parent_supplying_subtype():
    class Parent(TestComponentWithoutRun):
        def run(self) -> MyTrainingData:
            pass

    class MyComponent(TestComponentWithoutRun):
        def run(self, training_data: TrainingData) -> TrainingData:
            pass

    schema = create_test_schema(
        uses=MyComponent, parent=Parent, needs={"training_data": "parent"},
    )

    validation.validate(schema, language=None, is_train_graph=True)


def test_child_accepting_any_type_from_parent():
    class Parent(TestComponentWithoutRun):
        def run(self) -> MyTrainingData:
            pass

    class MyComponent(TestComponentWithoutRun):
        def run(self, training_data: Any) -> TrainingData:
            pass

    schema = create_test_schema(
        uses=MyComponent, parent=Parent, needs={"training_data": "parent"},
    )

    validation.validate(schema, language=None, is_train_graph=True)
