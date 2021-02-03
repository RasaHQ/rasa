import rasa.nlu

import pytest

import rasa.core.interpreter
from rasa.core.interpreter import RasaNLUHttpInterpreter, RasaNLUInterpreter
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.model import get_model_subdirectories, get_model
from rasa.nlu.model import Interpreter
from rasa.utils.endpoints import EndpointConfig


@pytest.mark.parametrize(
    "metadata",
    [
        {"rasa_version": "0.11.0"},
        {"rasa_version": "0.10.2"},
        {"rasa_version": "0.12.0a1"},
        {"rasa_version": "0.12.2"},
        {"rasa_version": "0.12.3"},
        {"rasa_version": "0.13.3"},
        {"rasa_version": "0.13.4"},
        {"rasa_version": "0.13.5"},
        {"rasa_version": "0.14.0a1"},
        {"rasa_version": "0.14.0"},
        {"rasa_version": "0.14.1"},
        {"rasa_version": "0.14.2"},
        {"rasa_version": "0.14.3"},
        {"rasa_version": "0.14.4"},
        {"rasa_version": "0.15.0a1"},
        {"rasa_version": "1.0.0a1"},
        {"rasa_version": "1.5.0"},
    ],
)
def test_model_is_not_compatible(metadata):
    with pytest.raises(rasa.nlu.model.UnsupportedModelError):
        Interpreter.ensure_model_compatibility(metadata)


@pytest.mark.parametrize("metadata", [{"rasa_version": rasa.__version__}])
def test_model_is_compatible(metadata):
    # should not raise an exception
    assert Interpreter.ensure_model_compatibility(metadata) is None


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "obj": "not-existing",
            "endpoint": EndpointConfig(url="http://localhost:8080/"),
            "type": RasaNLUHttpInterpreter,
        },
        {
            "obj": "trained_nlu_model",
            "endpoint": EndpointConfig(url="http://localhost:8080/"),
            "type": RasaNLUHttpInterpreter,
        },
        {"obj": "trained_nlu_model", "endpoint": None, "type": RasaNLUInterpreter},
        {"obj": "not-existing", "endpoint": None, "type": RegexInterpreter},
    ],
)
def test_create_interpreter(parameters, trained_nlu_model):
    obj = parameters["obj"]
    if obj == "trained_nlu_model":
        _, obj = get_model_subdirectories(get_model(trained_nlu_model))

    interpreter = rasa.core.interpreter.create_interpreter(
        parameters["endpoint"] or obj
    )

    assert isinstance(interpreter, parameters["type"])
