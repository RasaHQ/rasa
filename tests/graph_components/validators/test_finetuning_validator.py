import os
from pathlib import Path
import copy
from typing import Callable, List, Optional, Text, Dict, Any

from _pytest.monkeypatch import MonkeyPatch
import pytest


from rasa.engine.graph import ExecutionContext, GraphComponent, GraphSchema, SchemaNode
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.validators.finetuning_validator import FinetuningValidator
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.core.policies.rule_policy import RulePolicy
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
)
from rasa.shared.core.domain import KEY_RESPONSES, Domain
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.importers.importer import NluDataImporter, TrainingDataImporter
import rasa.shared.utils.io
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.constants import ACTION_NAME, INTENT, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData


@pytest.fixture
def default_resource() -> Resource:
    return Resource("FineTuningValidator")


ValidationMethodType = Callable[
    [TrainingDataImporter, Dict[Text, Any]], TrainingDataImporter
]


@pytest.fixture
def get_finetuning_validator(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    default_resource: Resource,
) -> Callable[[bool, bool, Dict[Text, Any], GraphSchema], FinetuningValidator]:
    def inner(
        finetuning: bool,
        load: bool,
        config: Dict[Text, Any],
        graph_schema: Optional[GraphSchema] = None,
    ) -> FinetuningValidator:
        if load:
            constructor = FinetuningValidator.load
        else:
            constructor = FinetuningValidator.create
        if finetuning:
            default_execution_context.is_finetuning = finetuning
        if graph_schema is not None:
            default_execution_context.graph_schema = graph_schema
        return constructor(
            config={**FinetuningValidator.get_default_config(), **config},
            execution_context=default_execution_context,
            model_storage=default_model_storage,
            resource=default_resource,
        )

    return inner


@pytest.fixture
def get_validation_method(
    get_finetuning_validator: Callable[[bool, bool], FinetuningValidator]
) -> Callable[[bool, bool, bool, bool, GraphSchema], ValidationMethodType]:
    def inner(
        finetuning: bool,
        load: bool,
        nlu: bool,
        core: bool,
        graph_schema: Optional[GraphSchema] = None,
    ) -> ValidationMethodType:
        validator = get_finetuning_validator(
            finetuning=finetuning,
            load=load,
            config={"validate_core": core, "validate_nlu": nlu},
            graph_schema=graph_schema,
        )

        return validator.validate

    return inner


class DummyNLUDataImporter(NluDataImporter):
    def __init__(self, messages: List[Message]) -> None:
        self.training_data = TrainingData(training_examples=messages)

    def get_config(self) -> Dict:
        return {}

    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        return self.training_data


class EmptyDataImporter(DummyNLUDataImporter):
    def __init__(self) -> None:
        super().__init__([])


def _project_files(
    project: Text,
    config_file: Text = DEFAULT_CONFIG_PATH,
    domain: Text = DEFAULT_DOMAIN_PATH,
    training_files: Text = DEFAULT_DATA_PATH,
) -> TrainingDataImporter:
    paths = {
        "config_file": config_file,
        "domain_path": domain,
        "training_data_paths": training_files,
    }
    paths = {
        k: v if v is None or Path(v).is_absolute() else os.path.join(project, v)
        for k, v in paths.items()
    }
    paths["training_data_paths"] = [paths["training_data_paths"]]

    return RasaFileImporter(**paths)


@pytest.mark.parametrize("nlu, core", [(True, False), (False, True), (True, True)])
def test_validate_after_changing_response_text_in_domain(
    get_validation_method: Callable[..., ValidationMethodType],
    project: Text,
    nlu: bool,
    core: bool,
):
    # training
    importer = _project_files(project)
    old_domain = importer.get_domain()

    validate = get_validation_method(finetuning=False, load=False, core=core, nlu=nlu)
    validate(importer=importer)

    # Change NLG content but keep actions the same
    domain_with_changed_nlg = old_domain.as_dict()
    domain_with_changed_nlg[KEY_RESPONSES]["utter_greet"].append({"text": "hi"})
    domain_with_changed_nlg = Domain.from_dict(domain_with_changed_nlg)
    importer.get_domain = lambda: domain_with_changed_nlg

    # finetuning
    loaded_validate = get_validation_method(
        finetuning=False, load=True, core=core, nlu=nlu
    )
    assert importer.get_domain() != old_domain
    loaded_validate(importer=importer)


@pytest.mark.parametrize("nlu, core", [(True, False), (False, True), (True, True)])
def test_validate_after_adding_action_to_domain(
    get_validation_method: Callable[..., ValidationMethodType],
    project: Text,
    nlu: bool,
    core: bool,
):
    # training
    importer = _project_files(project)
    old_domain = importer.get_domain()

    validate = get_validation_method(finetuning=False, load=False, core=core, nlu=nlu)
    validate(importer=importer)

    # Add another action - via the response key
    domain_with_new_action = old_domain.as_dict()
    domain_with_new_action[KEY_RESPONSES]["utter_new"] = [{"text": "hi"}]
    domain_with_new_action = Domain.from_dict(domain_with_new_action)
    importer.get_domain = lambda: domain_with_new_action

    # finetuning
    loaded_validate = get_validation_method(
        finetuning=True, load=True, core=core, nlu=nlu
    )
    assert importer.get_domain() != old_domain
    if core:
        with pytest.raises(InvalidConfigException):
            loaded_validate(importer=importer)
    else:
        loaded_validate(importer=importer)


def _get_example_schema(num_epochs: int = 5, other_parameter: int = 10) -> GraphSchema:
    example_configs = [
        {
            "epochs": num_epochs,
            "other-parameter": other_parameter,
            "some-parameter": "bla",
        },
        {"epochs": num_epochs, "yet-other-parameter": 344},
        {"no-epochs-defined-here": None},
    ]
    return GraphSchema(
        nodes={
            f"node-{idx}": SchemaNode(
                needs={}, uses=GraphComponent, constructor_name="", fn="", config=config
            )
            for idx, config in enumerate(example_configs)
        }
    )


@pytest.mark.parametrize("nlu, core", [(True, False), (False, True), (True, True)])
def test_validate_after_changing_epochs_in_config(
    get_validation_method: Callable[..., ValidationMethodType], nlu: bool, core: bool
):
    # training
    schema1 = _get_example_schema(num_epochs=5)
    validate = get_validation_method(
        finetuning=False, load=False, nlu=nlu, core=core, graph_schema=schema1
    )
    validate(importer=EmptyDataImporter())

    # change schema - replace all epoch settings by a different value
    schema2 = _get_example_schema(num_epochs=5)
    for node in schema2.nodes.values():
        node.constructor_name = "other"

    # finetuning - does not complain
    loaded_validate = get_validation_method(
        finetuning=True, load=True, nlu=nlu, core=core, graph_schema=schema2
    )
    loaded_validate(importer=EmptyDataImporter())


@pytest.mark.parametrize("nlu, core", [(True, False), (False, True), (True, True)])
def test_validate_after_changing_constructor(
    get_validation_method: Callable[..., ValidationMethodType], nlu: bool, core: bool
):
    # training
    schema1 = _get_example_schema(num_epochs=5)
    validate = get_validation_method(
        finetuning=False, load=False, nlu=nlu, core=core, graph_schema=schema1
    )
    validate(importer=EmptyDataImporter())

    # change schema - replace all epoch settings by a different value
    schema2 = _get_example_schema(num_epochs=10)

    # finetuning - does not complain
    loaded_validate = get_validation_method(
        finetuning=True, load=True, nlu=nlu, core=core, graph_schema=schema2
    )
    loaded_validate(importer=EmptyDataImporter())


@pytest.mark.parametrize("nlu, core", [(True, False), (False, True), (True, True)])
def test_validate_after_removing_node_from_schema(
    get_validation_method: Callable[..., ValidationMethodType], nlu: bool, core: bool
):
    # training
    schema1 = _get_example_schema(num_epochs=5)
    validate = get_validation_method(
        finetuning=False, load=False, nlu=nlu, core=core, graph_schema=schema1
    )
    validate(importer=EmptyDataImporter())

    # change schema - remove a node
    schema2 = copy.deepcopy(schema1)
    schema2.nodes.pop(next(iter(schema2.nodes.keys())))

    # finetuning raises - doesn't matter if it's nlu/core/both
    loaded_validate = get_validation_method(
        finetuning=True, load=True, nlu=nlu, core=core, graph_schema=schema2
    )
    with pytest.raises(InvalidConfigException):
        loaded_validate(importer=EmptyDataImporter())


@pytest.mark.parametrize("nlu, core", [(True, False), (False, True), (True, True)])
def test_validate_after_adding_node_to_schema(
    get_validation_method: Callable[..., ValidationMethodType], nlu: bool, core: bool
):
    # training
    schema1 = _get_example_schema()
    schema2 = copy.deepcopy(schema1)
    schema2.nodes.pop(next(iter(schema2.nodes.keys())))

    validate = get_validation_method(
        finetuning=False, load=False, nlu=nlu, core=core, graph_schema=schema2
    )
    validate(importer=EmptyDataImporter())

    # change schema - continue with the schema with one more node than before
    assert len(schema1.nodes) > len(schema2.nodes)

    # finetuning raises -  doesn't matter if it's nlu/core/both
    loaded_validate = get_validation_method(
        finetuning=True, load=True, nlu=nlu, core=core, graph_schema=schema1
    )
    with pytest.raises(InvalidConfigException):
        loaded_validate(importer=EmptyDataImporter())


@pytest.mark.parametrize(
    "nlu, core, what",
    [
        (nlu, core, what)
        for what in ["uses", "needs", "fn", "config"]
        for nlu, core in [(True, False), (False, True), (True, True)]
    ],
)
def test_validate_after_replacing_something_in_schema(
    get_validation_method: Callable[..., ValidationMethodType],
    nlu: bool,
    core: bool,
    what: Text,
):
    # training
    schema1 = _get_example_schema()
    validate = get_validation_method(
        finetuning=False, load=False, nlu=nlu, core=core, graph_schema=schema1
    )
    validate(importer=EmptyDataImporter())

    # change schema
    schema2 = copy.deepcopy(schema1)
    schema_node = schema2.nodes["node-0"]
    if what == "uses":
        schema_node.uses = WhitespaceTokenizer
    elif what == "fn":
        schema_node.fn = "a-new-function"
    elif what == "needs":
        schema_node.needs = {"something-new": "node-1"}
    elif what == "config":
        schema_node.config["other-parameter"] = "some-new-value"
    else:
        assert False, "Please fix this test."

    # finetuning raises -  doesn't matter if it's nlu/core/both
    loaded_validate = get_validation_method(
        finetuning=True, load=True, nlu=nlu, core=core, graph_schema=schema2
    )
    with pytest.raises(InvalidConfigException):
        loaded_validate(importer=EmptyDataImporter())


@pytest.mark.parametrize("nlu, core", [(True, False), (False, True), (True, True)])
def test_validate_after_adding_adding_default_parameter(
    get_validation_method: Callable[..., ValidationMethodType], nlu: bool, core: bool
):
    # create a schema and rely on rasa to fill in defaults later
    schema1 = _get_example_schema()
    schema1.nodes["nlu-node"] = SchemaNode(
        needs={}, uses=WhitespaceTokenizer, constructor_name="", fn="", config={}
    )
    schema1.nodes["core-node"] = SchemaNode(
        needs={}, uses=RulePolicy, constructor_name="", fn="", config={}
    )

    # training
    validate = get_validation_method(
        finetuning=False, load=False, nlu=nlu, core=core, graph_schema=schema1
    )
    validate(importer=EmptyDataImporter())

    # same schema -- we just explicitly pass default values
    schema2 = copy.deepcopy(schema1)
    schema2.nodes["nlu-node"] = SchemaNode(
        needs={},
        uses=WhitespaceTokenizer,
        constructor_name="",
        fn="",
        config=WhitespaceTokenizer.get_default_config(),
    )
    schema2.nodes["core-node"] = SchemaNode(
        needs={},
        uses=RulePolicy,
        constructor_name="",
        fn="",
        config=RulePolicy.get_default_config(),
    )

    # finetuning *does not raise*
    loaded_validate = get_validation_method(
        finetuning=True, load=True, nlu=nlu, core=core, graph_schema=schema2
    )
    loaded_validate(importer=EmptyDataImporter())


@pytest.mark.parametrize(
    "nlu, core,key",
    [
        (nlu, core, key)
        for nlu, core in [(True, False), (False, True), (True, True)]
        for key in [INTENT, ACTION_NAME]
    ],
)
def test_validate_after_removing_or_adding_intent_or_action_name(
    get_validation_method: Callable[..., ValidationMethodType],
    nlu: bool,
    core: bool,
    key: Text,
):
    messages = [Message(data={key: "item-1"}), Message(data={key: "item-2"})]
    message_with_new_item = Message(data={key: "item-3"})

    # training
    importer = DummyNLUDataImporter(messages)
    validate = get_validation_method(finetuning=False, load=False, nlu=nlu, core=core)
    validate(importer=importer)

    # load validate method in finetuning mode
    validate = get_validation_method(finetuning=True, load=True, nlu=nlu, core=core)

    # ... apply with something suddenly missing
    importer2 = DummyNLUDataImporter(messages[1:])
    if nlu:
        with pytest.raises(InvalidConfigException):
            validate(importer=importer2)
    else:
        validate(importer=importer2)

    # ... apply with additional item
    importer3 = DummyNLUDataImporter(messages + [message_with_new_item])
    if nlu:
        with pytest.raises(InvalidConfigException):
            validate(importer=importer3)
    else:
        validate(importer=importer3)


@pytest.mark.parametrize(
    "nlu, core,key",
    [
        (nlu, core, key)
        for nlu, core in [(True, False), (False, True), (True, True)]
        for key in [INTENT, ACTION_NAME]
    ],
)
def test_validate_with_different_examples_for_intent_or_action_name(
    get_validation_method: Callable[..., ValidationMethodType],
    nlu: bool,
    core: bool,
    key: Text,
):
    messages = [
        Message(data={key: "item-1", TEXT: "a"}),
        Message(data={key: "item-2", TEXT: "b"}),
    ]

    # training
    importer = DummyNLUDataImporter(messages)
    validate = get_validation_method(finetuning=False, load=False, nlu=nlu, core=core)
    validate(importer=importer)

    # load validate method in finetuning mode
    validate = get_validation_method(finetuning=True, load=True, nlu=nlu, core=core)

    # ... apply with different messages
    messages = [
        Message(data={key: "item-1", TEXT: "c"}),
        Message(data={key: "item-1", TEXT: "d"}),
        Message(data={key: "item-2", TEXT: "e"}),
        Message(data={key: "item-2", TEXT: "f"}),
    ]
    importer2 = DummyNLUDataImporter(messages)
    # does not complain:
    validate(importer=importer2)


@pytest.mark.parametrize(
    "nlu, core, min_compatible_version, old_version, can_tune",
    [
        (nlu, core, old_version, min_compatible_version, can_tune)
        for nlu, core in [(True, False), (False, True), (True, True)]
        for old_version, min_compatible_version, can_tune in [
            ("2.1.0", "2.1.0", True),
            ("2.0.0", "2.1.0", True),
            ("2.1.0", "2.0.0", False),
        ]
    ],
)
def test_validate_with_other_version(
    monkeypatch: MonkeyPatch,
    get_validation_method: Callable[..., ValidationMethodType],
    nlu: bool,
    core: bool,
    min_compatible_version: Text,
    old_version: Text,
    can_tune: bool,
):
    monkeypatch.setattr(rasa, "__version__", old_version)
    monkeypatch.setattr(
        rasa.graph_components.validators.finetuning_validator,
        "MINIMUM_COMPATIBLE_VERSION",
        min_compatible_version,
    )

    # training
    importer = DummyNLUDataImporter([Message(data={INTENT: "dummy"})])
    validate = get_validation_method(finetuning=False, load=False, nlu=nlu, core=core)
    validate(importer=importer)

    # finetuning
    validate = get_validation_method(finetuning=True, load=True, nlu=nlu, core=core)
    if not can_tune:
        with pytest.raises(InvalidConfigException):
            validate(importer=importer)
    else:
        validate(importer=importer)


@pytest.mark.parametrize("nlu, core", [(True, False), (False, True), (True, True)])
def test_validate_with_finetuning_fails_without_training(
    get_validation_method: Callable[..., ValidationMethodType], nlu: bool, core: bool
):
    validate = get_validation_method(finetuning=True, load=False, nlu=nlu, core=core)
    with pytest.raises(InvalidConfigException):
        validate(importer=EmptyDataImporter())


def test_loading_without_persisting(
    get_finetuning_validator: Callable[
        [bool, bool, Dict[Text, bool]], FinetuningValidator
    ]
):
    with pytest.raises(ValueError):
        get_finetuning_validator(finetuning=False, load=True, config={})
