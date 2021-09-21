import os
from pathlib import Path
import copy
from typing import Callable, List, Optional, Text, Dict, Any
import functools

from _pytest.monkeypatch import MonkeyPatch
import pytest


from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.validators.finetuning_validator import FineTuningValidator
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


ValidationMethodType = Callable[[TrainingDataImporter, Dict[Text, Any]], None]


@pytest.fixture
def get_finetuning_validator(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    default_resource: Resource,
) -> Callable[[bool, bool], FineTuningValidator]:
    def inner(finetuning: bool, load: bool,) -> FineTuningValidator:
        if load:
            constructor = FineTuningValidator.load
        else:
            constructor = FineTuningValidator.create
        if finetuning:
            default_execution_context.is_finetuning = finetuning
        return constructor(
            config=FineTuningValidator.get_default_config(),
            execution_context=default_execution_context,
            model_storage=default_model_storage,
            resource=default_resource,
        )

    return inner


@pytest.fixture
def get_validation_method(
    get_finetuning_validator: Callable[[bool, bool], FineTuningValidator],
) -> Callable[[bool, bool, bool, bool], ValidationMethodType]:
    def inner(
        finetuning: bool, load: bool, nlu: bool, core: bool
    ) -> ValidationMethodType:
        validator = get_finetuning_validator(finetuning=finetuning, load=load)
        if core and nlu:
            method = "validate"
        elif core:
            method = "validate_core_only"
        elif nlu:
            method = "validate_nlu_only"
        else:
            method = "validate"
        func = getattr(validator, method)
        if not core and not nlu:
            func = functools.partial(func, nlu=False, core=False)
        return func

    return inner


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


def _get_example_config() -> Dict[Text, Any]:
    return {
        "language": "en",
        "pipeline": [
            {"name": "WhitespaceTokenizer"},
            {"name": "RegexFeaturizer"},
            {"name": "LexicalSyntacticFeaturizer"},
            {"name": "CountVectorsFeaturizer"},
            {
                "name": "CountVectorsFeaturizer",
                "analyzer": "char_wb",
                "min_ngram": 1,
                "max_ngram": 4,
            },
            {"name": "DIETClassifier", "epochs": 100},
            {"name": "EntitySynonymMapper"},
            {"name": "ResponseSelector", "epochs": 100},
            {
                "name": "FallbackClassifier",
                "threshold": 0.3,
                "ambiguity_threshold": 0.1,
            },
        ],
        "policies": [
            {"name": "MemoizationPolicy"},
            {"name": "TEDPolicy", "max_history": 5, "epochs": 100},
            {"name": "RulePolicy"},
        ],
    }


def _create_importer_from_config(
    config: Dict[Text, Any], path: Path, config_file_name: Text
) -> TrainingDataImporter:
    config1_path = path / config_file_name
    rasa.shared.utils.io.write_yaml(config, config1_path, True)
    return TrainingDataImporter.load_from_config(str(config1_path))


@pytest.mark.parametrize("nlu, core", [(True, False), (False, True), (True, True)])
def test_validate_after_changing_epochs_in_config(
    tmp_path: Path,
    get_validation_method: Callable[..., ValidationMethodType],
    nlu: bool,
    core: bool,
):
    # training
    config1 = _get_example_config()
    importer = _create_importer_from_config(config1, tmp_path, "config1.yml")
    validate = get_validation_method(finetuning=False, load=False, nlu=nlu, core=core)
    validate(importer=importer)

    # Change Configuration - replace all epoch settings by a different value
    config2 = copy.deepcopy(config1)
    replacements = 0
    for key in config2:
        for sub_config in config2[key]:
            if "epochs" in sub_config:
                sub_config["epochs"] = sub_config["epochs"] + 5
                replacements += 1
    assert (
        replacements > 0
    ), 'Please update the test such that the config used here contains "epoch" keys.'
    importer2 = _create_importer_from_config(config2, tmp_path, "config2.yml")

    # finetuning
    loaded_validate = get_validation_method(
        finetuning=True, load=True, nlu=nlu, core=core
    )
    loaded_validate(importer=importer2)


@pytest.mark.parametrize("nlu, core", [(True, False), (False, True), (True, True)])
def test_validate_after_changing_nlu_config(
    tmp_path: Path,
    get_validation_method: Callable[..., ValidationMethodType],
    nlu: bool,
    core: bool,
):
    # training
    config1 = _get_example_config()
    importer = _create_importer_from_config(config1, tmp_path, "config1.yml")
    validate = get_validation_method(finetuning=False, load=False, nlu=nlu, core=core)
    validate(importer=importer)

    # Change Config - remove parts of NLU pipeline
    config3 = copy.deepcopy(config1)
    config3["pipeline"] = config3["pipeline"][:1]  # drop NLU (except tokenizer)
    importer3 = _create_importer_from_config(config3, tmp_path, "config3.yml")

    # finetuning
    loaded_validate = get_validation_method(
        finetuning=True, load=True, nlu=nlu, core=core
    )

    # does raise - doesn't matter if it's nlu/core/both
    with pytest.raises(InvalidConfigException):
        loaded_validate(importer=importer3)


@pytest.mark.parametrize("nlu, core", [(True, False), (False, True), (True, True)])
def test_validate_after_changing_core_config(
    tmp_path: Path,
    get_validation_method: Callable[..., ValidationMethodType],
    nlu: bool,
    core: bool,
):
    # training
    config1 = _get_example_config()
    importer1 = _create_importer_from_config(config1, tmp_path, "config1.yml")
    validate = get_validation_method(finetuning=False, load=False, nlu=nlu, core=core)
    validate(importer=importer1)

    # Change Config - remove parts of NLU pipeline
    config3 = copy.deepcopy(config1)
    assert len(
        config3["policies"]
    ), "Please update the test so that this config has some more policies."
    config3["policies"] = config3["policies"][:1]  # drop some Policies
    importer3 = _create_importer_from_config(config3, tmp_path, "config3.yml")

    # finetuning
    loaded_validate = get_validation_method(
        finetuning=True, load=True, nlu=nlu, core=core
    )

    # does raise - doesn't matter if it's nlu/core/both
    with pytest.raises(InvalidConfigException):
        loaded_validate(importer=importer3)


class DummyNluDataImporter(NluDataImporter):
    def __init__(self, messages: List[Message], config: Dict[Text, Any]) -> None:
        self.training_data = TrainingData(training_examples=messages)
        self.config = config

    def get_config(self) -> Dict:
        return self.config

    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        return self.training_data


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
    messages = [
        Message(data={key: "item-1"}),
        Message(data={key: "item-2"}),
    ]
    message_with_new_item = Message(data={key: "item-3"})

    # training
    config = _get_example_config()
    importer = DummyNluDataImporter(messages, config)
    validate = get_validation_method(finetuning=False, load=False, nlu=nlu, core=core)
    validate(importer=importer)

    # load validate method in finetuning mode
    validate = get_validation_method(finetuning=True, load=True, nlu=nlu, core=core)

    # ... apply with something suddenly missing
    importer2 = DummyNluDataImporter(messages[1:], config)
    if nlu:
        with pytest.raises(InvalidConfigException):
            validate(importer=importer2)
    else:
        validate(importer=importer2)

    # ... apply with additional item
    importer3 = DummyNluDataImporter(messages + [message_with_new_item], config)
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
    config = _get_example_config()
    importer = DummyNluDataImporter(messages, config)
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
    importer2 = DummyNluDataImporter(messages, config)
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
    config = _get_example_config()
    importer = DummyNluDataImporter([Message(data={INTENT: "dummy"})], config)
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
    tmp_path: Path,
    get_validation_method: Callable[..., ValidationMethodType],
    nlu: bool,
    core: bool,
):
    config1 = _get_example_config()
    importer1 = _create_importer_from_config(config1, tmp_path, "config1.yml")
    validate = get_validation_method(finetuning=True, load=False, nlu=nlu, core=core)
    with pytest.raises(InvalidConfigException):
        validate(importer=importer1)


def test_loading_without_persisting(
    get_finetuning_validator: Callable[[bool, bool], FineTuningValidator],
):
    with pytest.raises(ValueError):
        get_finetuning_validator(finetuning=False, load=True)
