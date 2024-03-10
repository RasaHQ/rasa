from typing import Any, Callable, Dict, Text, Tuple, Type, Optional, List
from collections import namedtuple
import itertools
from unittest.mock import Mock

import pytest
from rasa.core.policies.policy import PolicyPrediction

from rasa.engine import validation
from rasa.engine.exceptions import GraphSchemaValidationException
from rasa.engine.graph import (
    GraphComponent,
    ExecutionContext,
    GraphSchema,
    SchemaNode,
    GraphModelConfiguration,
)
from rasa.engine.constants import PLACEHOLDER_IMPORTER
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.data import TrainingType
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.message import Message
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


class TestComponentWithRunAndParam(TestComponentWithoutRun):
    def run(self, training_data: TrainingData) -> TrainingData:
        pass


class TestNLUTarget(TestComponentWithoutRun):
    def run(self) -> List[Message]:
        pass


class TestCoreTarget(TestComponentWithoutRun):
    def run(self) -> PolicyPrediction:
        pass


class TestComponentWithClsTypeHints(GraphComponent):
    @classmethod
    def create(
        cls: "TestComponentWithClsTypeHints",
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls()


DEFAULT_PREDICT_SCHEMA = GraphSchema(
    {
        "nlu_target": SchemaNode(
            needs={},
            uses=TestNLUTarget,
            eager=True,
            constructor_name="load",
            fn="run",
            config={},
        )
    }
)


def create_test_schema(
    uses: Type,  # The unspecified type is on purpose to enable testing of invalid cases
    constructor_name: Text = "create",
    run_fn: Text = "run",
    needs: Optional[Dict[Text, Text]] = None,
    eager: bool = True,
    parent: Optional[Type[GraphComponent]] = None,
    language: Optional[Text] = None,
    is_train_graph: bool = True,
) -> GraphModelConfiguration:

    parent_node = {}
    if parent:
        parent_node = {
            "parent": SchemaNode(
                needs={}, uses=parent, constructor_name="create", fn="run", config={}
            )
        }

    train_schema = GraphSchema({})
    predict_schema = DEFAULT_PREDICT_SCHEMA
    # noinspection PyTypeChecker
    schema = GraphSchema(
        {
            "my_node": SchemaNode(
                needs=needs or {},
                uses=uses,
                eager=eager,
                constructor_name=constructor_name,
                fn=run_fn,
                config={},
            ),
            **DEFAULT_PREDICT_SCHEMA.nodes,
            **parent_node,
        }
    )

    if is_train_graph:
        train_schema = schema
    else:
        predict_schema = schema

    return GraphModelConfiguration(
        train_schema=train_schema,
        predict_schema=predict_schema,
        training_type=TrainingType.BOTH,
        assistant_id="test_assistant",
        core_target=None,
        nlu_target="nlu_target",
        language=language,
    )


def test_graph_component_is_no_graph_component():
    class MyComponent:
        def other(self) -> TrainingData:
            pass

    graph_config = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="implement .+ interface"):
        validation.validate(graph_config)


def test_graph_component_fn_does_not_exist():
    graph_config = create_test_schema(uses=TestComponentWithRun, run_fn="some_fn")

    with pytest.raises(
        GraphSchemaValidationException, match="required method 'some_fn'"
    ):
        validation.validate(graph_config)


def test_graph_output_is_not_fingerprintable_int():
    class MyComponent(TestComponentWithoutRun):
        def run(self) -> int:
            pass

    graph_config = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="fingerprintable"):
        validation.validate(graph_config)


def test_predict_graph_output_is_not_fingerprintable():
    class MyComponent(TestComponentWithoutRun):
        def run(self) -> int:
            pass

    graph_config = create_test_schema(uses=MyComponent, is_train_graph=False)

    validation.validate(graph_config)


def test_graph_output_is_not_fingerprintable_any():
    class MyComponent(TestComponentWithoutRun):
        def run(self) -> Any:
            pass

    graph_config = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="fingerprintable"):
        validation.validate(graph_config)


def test_graph_output_is_not_fingerprintable_None():
    class MyComponent(TestComponentWithoutRun):
        def run(self) -> None:
            pass

    graph_config = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="fingerprintable"):
        validation.validate(graph_config)


def test_graph_with_forward_referenced_output_type():
    class MyComponent(TestComponentWithoutRun):
        # The non imported type annotation is on purpose so we can provoke a error in
        # the test
        def run(self) -> "UserUttered":  # noqa: F821
            pass

    graph_config = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="forward reference"):
        validation.validate(graph_config)


def test_graph_output_missing_type_annotation():
    class MyComponent(TestComponentWithoutRun):
        def run(self):
            pass

    graph_config = create_test_schema(uses=MyComponent)

    with pytest.raises(
        GraphSchemaValidationException, match="does not have a type annotation"
    ):
        validation.validate(graph_config)


def test_graph_with_fingerprintable_output():
    class MyComponent(TestComponentWithoutRun):
        def run(self) -> TrainingData:
            pass

    graph_config = create_test_schema(uses=MyComponent)

    validation.validate(graph_config)


class MyTrainingData(TrainingData):
    pass


def test_graph_with_fingerprintable_output_subclass():
    class MyComponent(TestComponentWithoutRun):
        def run(self) -> MyTrainingData:
            pass

    graph_config = create_test_schema(uses=MyComponent)

    validation.validate(graph_config)


def test_graph_constructor_missing():
    class MyComponent(TestComponentWithoutRun):
        def run(self) -> TrainingData:
            pass

    graph_config = create_test_schema(uses=MyComponent, constructor_name="invalid")

    with pytest.raises(
        GraphSchemaValidationException, match="required method 'invalid'"
    ):
        validation.validate(graph_config)


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

    graph_config = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="incompatible type"):
        validation.validate(graph_config)


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

    graph_config = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="incompatible type"):
        validation.validate(graph_config)


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

    graph_config = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="incompatible type"):
        validation.validate(graph_config)


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

    graph_config = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="incompatible type"):
        validation.validate(graph_config)


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

    graph_config = create_test_schema(uses=MyComponent, language=current_language)

    with pytest.raises(
        GraphSchemaValidationException, match="does not support .* language"
    ):
        validation.validate(graph_config)


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

    graph_config = create_test_schema(uses=MyComponent, language=current_language)

    validation.validate(graph_config)


@pytest.mark.parametrize(
    "current_language, not_supported_languages", [("de", ["de", "en"]), ("en", ["en"])]
)
def test_graph_constructor_execution_exclusive_list_not_supported_language(
    current_language: Text, not_supported_languages: Optional[List[Text]]
):
    class MyComponent(TestComponentWithRun):
        @staticmethod
        def not_supported_languages() -> Optional[List[Text]]:
            return not_supported_languages

    graph_config = create_test_schema(
        uses=MyComponent, language=current_language, is_train_graph=False
    )

    with pytest.raises(
        GraphSchemaValidationException, match="does not support .* language"
    ):
        validation.validate(graph_config)


@pytest.mark.parametrize(
    "current_language, not_supported_languages",
    [(None, None), ("en", ["zh"]), ("zh", None), (None, ["de"])],
)
def test_graph_constructor_execution_exclusive_list_supported_language(
    current_language: Optional[Text], not_supported_languages: Optional[List[Text]]
):
    class MyComponent(TestComponentWithRun):
        @staticmethod
        def not_supported_languages() -> Optional[List[Text]]:
            return not_supported_languages

    graph_config = create_test_schema(
        uses=MyComponent, language=current_language, is_train_graph=False
    )

    validation.validate(graph_config)


@pytest.mark.parametrize(
    "required_packages", [["pytorch"], ["tensorflow", "kubernetes"]]
)
def test_graph_missing_package_requirements(required_packages: List[Text]):
    class MyComponent(TestComponentWithRun):
        @staticmethod
        def required_packages() -> List[Text]:
            """Any extra python dependencies required for this component to run."""
            return required_packages

    graph_config = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="not installed"):
        validation.validate(graph_config)


@pytest.mark.parametrize("required_packages", [["tensorflow"], ["tensorflow", "numpy"]])
def test_graph_satisfied_package_requirements(required_packages: List[Text]):
    class MyComponent(TestComponentWithRun):
        @staticmethod
        def required_packages() -> List[Text]:
            """Any extra python dependencies required for this component to run."""
            return required_packages

    graph_config = create_test_schema(uses=MyComponent)

    validation.validate(graph_config)


def test_run_param_not_satisfied():
    class MyComponent(TestComponentWithoutRun):
        def run(self, some_param: TrainingData) -> TrainingData:
            pass

    graph_config = create_test_schema(uses=MyComponent)

    with pytest.raises(GraphSchemaValidationException, match="needs the param"):
        validation.validate(graph_config)


def test_run_param_satifisfied_due_to_default():
    class MyComponent(TestComponentWithoutRun):
        def run(self, some_param: TrainingData = TrainingData()) -> TrainingData:
            pass

    graph_config = create_test_schema(uses=MyComponent)

    validation.validate(graph_config)


def test_too_many_supplied_params():
    graph_config = create_test_schema(
        uses=TestComponentWithRun, needs={"some_param": "parent"}
    )

    with pytest.raises(
        GraphSchemaValidationException, match="does not accept a parameter"
    ):
        validation.validate(graph_config)


def test_too_many_supplied_params_but_kwargs():
    class MyComponent(TestComponentWithoutRun):
        def run(self, **kwargs: Any) -> TrainingData:
            pass

    graph_config = create_test_schema(
        uses=MyComponent, needs={"some_param": "parent"}, parent=TestComponentWithRun
    )

    validation.validate(graph_config)


def test_run_fn_with_variable_length_positional_param():
    class MyComponent(TestComponentWithoutRun):
        def run(self, *args: Any, some_param: TrainingData) -> TrainingData:
            pass

    graph_config = create_test_schema(
        uses=MyComponent, needs={"some_param": "parent"}, parent=TestComponentWithRun
    )

    validation.validate(graph_config)


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

    graph_config = create_test_schema(
        uses=MyComponent,
        needs={"some_param": "parent"},
        eager=False,
        constructor_name="load",
        parent=TestComponentWithRun,
    )

    validation.validate(graph_config)


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

    graph_config = create_test_schema(
        uses=MyComponent,
        needs={"some_param": "parent"},
        eager=True,
        constructor_name="load",
    )

    with pytest.raises(
        GraphSchemaValidationException, match="which is used during training"
    ):
        validation.validate(graph_config)


@pytest.mark.parametrize(
    "eager, error_message", [(True, "during training"), (False, "needs the param")]
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

    graph_config = create_test_schema(
        uses=MyComponent, eager=eager, constructor_name="load"
    )

    with pytest.raises(GraphSchemaValidationException, match=error_message):
        validation.validate(graph_config)


def test_parent_is_missing():
    graph_config = create_test_schema(
        uses=TestComponentWithRunAndParam,
        needs={"training_data": "not existing parent"},
    )

    with pytest.raises(
        GraphSchemaValidationException, match="The component is missing from"
    ):
        validation.validate(graph_config)


def test_parent_supplying_wrong_type():
    class MyUnreliableParent(TestComponentWithoutRun):
        def run(self) -> Domain:
            pass

    graph_config = create_test_schema(
        uses=TestComponentWithRunAndParam,
        parent=MyUnreliableParent,
        needs={"training_data": "parent"},
    )

    with pytest.raises(
        GraphSchemaValidationException, match="expects an input of type"
    ):
        validation.validate(graph_config)


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

    graph_config = create_test_schema(
        uses=MyComponent,
        eager=False,
        constructor_name="load",
        parent=MyUnreliableParent,
        needs={"some_param": "parent"},
    )

    with pytest.raises(
        GraphSchemaValidationException, match="expects an input of type"
    ):
        validation.validate(graph_config)


def test_parent_supplying_subtype():
    class Parent(TestComponentWithoutRun):
        def run(self) -> MyTrainingData:
            pass

    class MyComponent(TestComponentWithoutRun):
        def run(self, training_data: TrainingData) -> TrainingData:
            pass

    graph_config = create_test_schema(
        uses=MyComponent, parent=Parent, needs={"training_data": "parent"}
    )

    validation.validate(graph_config)


def test_child_accepting_any_type_from_parent():
    class Parent(TestComponentWithoutRun):
        def run(self) -> MyTrainingData:
            pass

    class MyComponent(TestComponentWithoutRun):
        def run(self, training_data: Any) -> TrainingData:
            pass

    graph_config = create_test_schema(
        uses=MyComponent, parent=Parent, needs={"training_data": "parent"}
    )

    validation.validate(graph_config)


@pytest.mark.parametrize("is_train_graph", [True, False])
def test_cycle(is_train_graph: bool):
    class MyTestComponent(TestComponentWithoutRun):
        def run(self, training_data: TrainingData) -> TrainingData:
            pass

    train_schema = GraphSchema({})
    predict_schema = DEFAULT_PREDICT_SCHEMA

    schema = GraphSchema(
        {
            "A": SchemaNode(
                needs={"training_data": "B"},
                uses=MyTestComponent,
                eager=True,
                constructor_name="create",
                fn="run",
                is_target=True,
                config={},
            ),
            "B": SchemaNode(
                needs={"training_data": "C"},
                uses=MyTestComponent,
                eager=True,
                constructor_name="create",
                fn="run",
                config={},
            ),
            "C": SchemaNode(
                needs={"training_data": "A"},
                uses=MyTestComponent,
                eager=True,
                constructor_name="create",
                fn="run",
                config={},
            ),
        }
    )

    if is_train_graph:
        train_schema = schema
    else:
        predict_schema = schema

    with pytest.raises(GraphSchemaValidationException, match="Cycles"):
        validation.validate(
            GraphModelConfiguration(
                train_schema=train_schema,
                predict_schema=predict_schema,
                training_type=TrainingType.BOTH,
                assistant_id="test_assistant",
                language=None,
                core_target=None,
                nlu_target="nlu_target",
            )
        )


def test_validation_with_placeholders():
    class MyTestComponent(TestComponentWithoutRun):
        def run(self, training_data: TrainingDataImporter) -> TrainingDataImporter:
            pass

    graph_config = GraphSchema(
        {
            "A": SchemaNode(
                needs={"training_data": "B"},
                uses=MyTestComponent,
                eager=True,
                constructor_name="create",
                fn="run",
                is_target=True,
                config={},
            ),
            "B": SchemaNode(
                needs={"training_data": PLACEHOLDER_IMPORTER},
                uses=MyTestComponent,
                eager=True,
                constructor_name="create",
                fn="run",
                config={},
            ),
        }
    )

    # Does not raise
    validation.validate(
        GraphModelConfiguration(
            train_schema=graph_config,
            predict_schema=DEFAULT_PREDICT_SCHEMA,
            training_type=TrainingType.BOTH,
            assistant_id="test_assistant",
            language=None,
            core_target=None,
            nlu_target="nlu_target",
        )
    )


def test_validation_with_missing_nlu_target():
    graph_config = GraphSchema(
        {
            "A": SchemaNode(
                needs={},
                uses=TestNLUTarget,
                eager=True,
                constructor_name="create",
                fn="run",
                config={},
            )
        }
    )

    with pytest.raises(
        GraphSchemaValidationException, match="no target for the 'nlu_target'"
    ):
        validation.validate(
            GraphModelConfiguration(
                train_schema=GraphSchema({}),
                predict_schema=graph_config,
                training_type=TrainingType.BOTH,
                assistant_id="test_assistant",
                language=None,
                core_target=None,
                nlu_target=None,
            )
        )


def test_validation_with_nlu_target_used_by_other_node():
    class NLUTargetConsumer(TestComponentWithoutRun):
        def run(self, nlu_target_output: List[Message]) -> List[Message]:
            pass

    graph_config = GraphSchema(
        {
            "A": SchemaNode(
                needs={},
                uses=TestNLUTarget,
                eager=True,
                constructor_name="create",
                fn="run",
                config={},
            ),
            "B": SchemaNode(
                needs={"nlu_target_output": "A"},
                uses=NLUTargetConsumer,
                eager=True,
                constructor_name="create",
                fn="run",
                config={},
            ),
        }
    )

    with pytest.raises(
        GraphSchemaValidationException, match="uses the NLU target 'A' as input"
    ):
        validation.validate(
            GraphModelConfiguration(
                train_schema=GraphSchema({}),
                predict_schema=graph_config,
                training_type=TrainingType.BOTH,
                assistant_id="test_assistant",
                language=None,
                core_target=None,
                nlu_target="A",
            )
        )


def test_validation_with_nlu_target_wrong_type():
    graph_config = GraphSchema(
        {
            "A": SchemaNode(
                needs={},
                uses=TestCoreTarget,
                eager=True,
                constructor_name="create",
                fn="run",
                config={},
            )
        }
    )

    with pytest.raises(GraphSchemaValidationException, match="invalid return type"):
        validation.validate(
            GraphModelConfiguration(
                train_schema=GraphSchema({}),
                predict_schema=graph_config,
                training_type=TrainingType.BOTH,
                assistant_id="test_assistant",
                language=None,
                core_target=None,
                nlu_target="A",
            )
        )


def test_validation_with_missing_core_target():
    graph_config = GraphSchema(
        {
            "A": SchemaNode(
                needs={},
                uses=TestNLUTarget,
                eager=True,
                constructor_name="create",
                fn="run",
                config={},
            )
        }
    )

    with pytest.raises(GraphSchemaValidationException, match="invalid Core target"):
        validation.validate(
            GraphModelConfiguration(
                train_schema=GraphSchema({}),
                predict_schema=graph_config,
                training_type=TrainingType.BOTH,
                assistant_id="test_assistant",
                language=None,
                core_target="B",
                nlu_target="A",
            )
        )


def test_validation_with_core_target_wrong_type():
    graph_config = GraphSchema(
        {
            "A": SchemaNode(
                needs={},
                uses=TestNLUTarget,
                eager=True,
                constructor_name="create",
                fn="run",
                config={},
            )
        }
    )

    with pytest.raises(
        GraphSchemaValidationException, match="Core model's .* invalid return type"
    ):
        validation.validate(
            GraphModelConfiguration(
                train_schema=GraphSchema({}),
                predict_schema=graph_config,
                training_type=TrainingType.BOTH,
                assistant_id="test_assistant",
                language=None,
                core_target="A",
                nlu_target="A",
            )
        )


def test_validation_with_core_target_used_by_other_node():
    class CoreTargetConsumer(TestComponentWithoutRun):
        def run(self, core_target_output: PolicyPrediction) -> PolicyPrediction:
            pass

    graph_config = GraphSchema(
        {
            "A": SchemaNode(
                needs={},
                uses=TestNLUTarget,
                eager=True,
                constructor_name="create",
                fn="run",
                config={},
            ),
            "B": SchemaNode(
                needs={},
                uses=TestCoreTarget,
                eager=True,
                constructor_name="create",
                fn="run",
                config={},
            ),
            "C": SchemaNode(
                needs={"core_target_output": "B"},
                uses=CoreTargetConsumer,
                eager=True,
                constructor_name="create",
                fn="run",
                config={},
            ),
        }
    )

    with pytest.raises(
        GraphSchemaValidationException, match="uses the Core target 'B' as input"
    ):
        validation.validate(
            GraphModelConfiguration(
                train_schema=GraphSchema({}),
                predict_schema=graph_config,
                training_type=TrainingType.BOTH,
                assistant_id="test_assistant",
                language=None,
                core_target="B",
                nlu_target="A",
            )
        )


def _create_run_function(num_args) -> Callable[..., TrainingData]:
    # Note: setting __annotations__ is not sufficient for the validation and
    # creating a function via types.FunctionType is cumbersome, so we just
    # explicitly create the function we need:
    if num_args == 0:

        def run() -> TrainingData:
            return TrainingData()

    elif num_args == 1:

        def run(param0: TrainingData) -> TrainingData:
            return TrainingData()

    elif num_args == 2:

        def run(param0: TrainingData, param1: TrainingData) -> TrainingData:
            return TrainingData()

    elif num_args == 3:

        def run(
            param0: TrainingData, param1: TrainingData, param2: TrainingData
        ) -> TrainingData:
            return TrainingData()

    else:
        assert False, f"This test doesn't work with num_args={num_args} ."
    return run


def _create_component_type_and_subtype_with_run_function(
    component_type_name: Text, needs: List[int]
) -> Tuple[Type[GraphComponent], Type[GraphComponent]]:
    main_type = type(
        component_type_name,
        (TestComponentWithoutRun,),
        {
            "run": _create_run_function(num_args=len(needs)),
            "create": lambda *args, **kwargs: None,
            "__init__": lambda: None,
        },
    )
    sub_type = type(f"subclass_of_{component_type_name}", (main_type,), {})
    return main_type, sub_type


def _create_graph_schema_from_requirements(
    node_needs_requires: List[Tuple[int, List[int], List[int]]],
    targets: List[int],
    use_subclass: bool,
) -> GraphSchema:
    # create some component types
    component_types = {
        node: _create_component_type_and_subtype_with_run_function(
            component_type_name=f"class_{node}", needs=needs
        )
        for node, needs, _ in node_needs_requires
    }

    # add required components
    for node, _, required_components in node_needs_requires:
        for component_type in component_types[node]:
            component_type.required_components = Mock(
                return_value=[
                    component_types[required][0] for required in required_components
                ]
            )

    # create graph schema
    graph_schema = GraphSchema(
        {
            f"node-{node}": SchemaNode(
                needs={
                    f"param{param}": f"node-{needed_node}"
                    for param, needed_node in enumerate(needs)
                },
                uses=component_types[node][use_subclass],  # use subclass if required
                fn="run",
                constructor_name="create",
                config={},
                is_target=node in targets,
            )
            for node, needs, _ in node_needs_requires
        }
    )
    graph_schema.nodes.update(DEFAULT_PREDICT_SCHEMA.nodes)
    return graph_schema


RequiredComponentsTestCase = namedtuple(
    "RequiredComponentsTestCase",
    {
        "node_needs_requires_tuples": List[Tuple[int, List[int], List[int]]],
        "targets": List[int],
        "num_unmet_requirements": int,
    },
)
REQUIRED_COMPONENT_TEST_CASES: List[RequiredComponentsTestCase] = [
    RequiredComponentsTestCase(
        node_needs_requires_tuples=[(1, [2], [2]), (2, [], [])],
        targets=[1],
        num_unmet_requirements=0,
    ),
    RequiredComponentsTestCase(
        node_needs_requires_tuples=[
            (1, [2, 3, 4], [2, 3, 4]),
            (2, [], []),
            (3, [], []),
            (4, [], []),
        ],
        targets=[1],
        num_unmet_requirements=0,
    ),
    RequiredComponentsTestCase(
        node_needs_requires_tuples=[
            (1, [3], [4]),
            (2, [3], [4]),
            (3, [4, 5], []),
            (4, [], []),
            (5, [6], []),
            (6, [], []),
        ],
        targets=[1, 3],
        num_unmet_requirements=0,
    ),
    RequiredComponentsTestCase(
        node_needs_requires_tuples=[(1, [], [2]), (2, [], [])],
        targets=[1],
        num_unmet_requirements=1,
    ),  # 2 is not reachable from 1
    RequiredComponentsTestCase(
        node_needs_requires_tuples=[
            (1, [3], [4]),
            (2, [3], [4]),
            (3, [4], [5]),
            (4, [], []),
            (5, [], []),
        ],
        targets=[1],
        num_unmet_requirements=1,  # 5 is not reachable from 3
    ),
    RequiredComponentsTestCase(
        node_needs_requires_tuples=[
            (1, [2], [3]),
            (2, [], [4]),
            (3, [], []),
            (4, [], []),
        ],
        targets=[1],
        num_unmet_requirements=2,
    ),  # 3 and 4 are not reachable from 1 and 2
]


@pytest.mark.parametrize(
    "test_case, is_train_graph, test_subclass",
    itertools.product(REQUIRED_COMPONENT_TEST_CASES, [True, False], [True, False]),
)
def test_validate_validates_required_components(
    test_case: List[RequiredComponentsTestCase],
    is_train_graph: bool,
    test_subclass: bool,
):
    train_schema = GraphSchema({})
    predict_schema = DEFAULT_PREDICT_SCHEMA
    graph_schema = _create_graph_schema_from_requirements(
        node_needs_requires=test_case.node_needs_requires_tuples,
        targets=test_case.targets,
        use_subclass=test_subclass,
    )

    if is_train_graph:
        train_schema = graph_schema
    else:
        predict_schema = graph_schema
    graph_config = GraphModelConfiguration(
        train_schema,
        predict_schema,
        TrainingType.BOTH,
        "test_assistant",
        None,
        None,
        "nlu_target",
    )

    num_unmet = test_case.num_unmet_requirements
    if num_unmet == 0:
        validation.validate(graph_config)
    else:
        message = f"{num_unmet} components are missing"
        with pytest.raises(GraphSchemaValidationException, match=message):
            validation.validate(graph_config)


@pytest.mark.parametrize(
    "test_case, test_subclass",
    itertools.product(REQUIRED_COMPONENT_TEST_CASES, [True, False]),
)
def test_validate_required_components(
    test_case: List[RequiredComponentsTestCase], test_subclass: bool
):
    graph_schema = _create_graph_schema_from_requirements(
        node_needs_requires=test_case.node_needs_requires_tuples,
        targets=test_case.targets,
        use_subclass=test_subclass,
    )
    num_unmet = test_case.num_unmet_requirements
    if num_unmet == 0:
        validation._validate_required_components(schema=graph_schema)
    else:
        message = f"{num_unmet} components are missing"
        with pytest.raises(GraphSchemaValidationException, match=message):
            validation._validate_required_components(schema=graph_schema)


@pytest.mark.parametrize(
    "test_case, test_subclass",
    itertools.product(
        [
            test_case
            for test_case in REQUIRED_COMPONENT_TEST_CASES
            if len(test_case.targets) == 1
        ],
        [True, False],
    ),
)
def test_recursively_validate_required_components(
    test_case: List[RequiredComponentsTestCase], test_subclass: bool
):
    graph_schema = _create_graph_schema_from_requirements(
        node_needs_requires=test_case.node_needs_requires_tuples,
        targets=test_case.targets,
        use_subclass=test_subclass,
    )
    num_unmet = test_case.num_unmet_requirements

    unmet_requirements, _ = validation._recursively_check_required_components(
        node_name=f"node-{test_case.targets[0]}", schema=graph_schema
    )
    assert len(unmet_requirements) == num_unmet


def test_graph_with_cls_type_hint():
    class MyComponent(TestComponentWithClsTypeHints):
        def run(self) -> MyTrainingData:
            pass

    graph_config = create_test_schema(uses=MyComponent)

    validation.validate(graph_config)
