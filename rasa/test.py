import asyncio
import logging
import tempfile
from typing import Text, Dict, Optional, List
import os

from rasa.constants import DEFAULT_RESULTS_PATH
from rasa.model import get_model, get_model_subdirectories, unpack_model
from rasa.cli.utils import minimal_kwargs

logger = logging.getLogger(__name__)


def test_compare(models: List[Text], stories: Text, output: Text):
    from rasa.core.test import compare, plot_curve
    import rasa.core.utils as core_utils

    model_directory = copy_models_to_compare(models)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(compare(model_directory, stories, output))

    story_n_path = os.path.join(model_directory, "num_stories.json")
    number_of_stories = core_utils.read_json_file(story_n_path)
    plot_curve(output, number_of_stories)


def test(
    model: Text,
    stories: Text,
    nlu_data: Text,
    endpoints: Optional[Text] = None,
    output: Text = DEFAULT_RESULTS_PATH,
    kwargs: Optional[Dict] = None,
):
    if kwargs is None:
        kwargs = {}

    test_core(model, stories, endpoints, output, **kwargs)
    test_nlu(model, nlu_data, kwargs)


def test_core(
    model: Optional[Text] = None,
    stories: Optional[Text] = None,
    endpoints: Optional[Text] = None,
    output: Text = DEFAULT_RESULTS_PATH,
    kwargs: Optional[Dict] = None,
):
    import rasa.core.test
    import rasa.core.utils as core_utils
    from rasa.nlu import utils as nlu_utils
    from rasa.model import get_model
    from rasa.core.interpreter import NaturalLanguageInterpreter
    from rasa.core.agent import Agent

    _endpoints = core_utils.AvailableEndpoints.read_endpoints(endpoints)

    if kwargs is None:
        kwargs = {}

    if output:
        nlu_utils.create_dir(output)

    loop = asyncio.get_event_loop()
    model_path = get_model(model)
    core_path, nlu_path = get_model_subdirectories(model_path)

    if os.path.exists(core_path) and os.path.exists(nlu_path):
        _interpreter = NaturalLanguageInterpreter.create(nlu_path, _endpoints.nlu)

        _agent = Agent.load(model_path, interpreter=_interpreter)

        kwargs = minimal_kwargs(kwargs, rasa.core.test, ["stories", "agent"])

        loop.run_until_complete(
            rasa.core.test(stories, _agent, out_directory=output, **kwargs)
        )
    else:
        logger.error(
            "Not able to test. Make sure both models, core and " "nlu, are available."
        )


def test_nlu(model: Optional[Text], nlu_data: Optional[Text], kwargs: Optional[Dict]):
    from rasa.nlu.test import run_evaluation

    unpacked_model = get_model(model)
    nlu_model = os.path.join(unpacked_model, "nlu")

    if os.path.exists(nlu_model):
        kwargs = minimal_kwargs(kwargs, run_evaluation, ["data_path", "model"])
        run_evaluation(nlu_data, nlu_model, **kwargs)


def test_nlu_with_cross_validation(config: Text, nlu: Text, folds: int = 3):
    import rasa.nlu.config
    import rasa.nlu.test as nlu_test

    nlu_config = rasa.nlu.config.load(config)
    data = rasa.nlu.training_data.load_data(nlu)
    data = nlu_test.drop_intents_below_freq(data, cutoff=5)
    results, entity_results = nlu_test.cross_validate(data, int(folds), nlu_config)
    logger.info("CV evaluation (n={})".format(folds))

    if any(results):
        logger.info("Intent evaluation results")
        nlu_test.return_results(results.train, "train")
        nlu_test.return_results(results.test, "test")
    if any(entity_results):
        logger.info("Entity evaluation results")
        nlu_test.return_entity_results(entity_results.train, "train")
        nlu_test.return_entity_results(entity_results.test, "test")


def copy_models_to_compare(models: List[str]) -> Text:
    models_dir = tempfile.mkdtemp()

    for i, model in enumerate(models):
        if os.path.exists(model) and os.path.isfile(model):
            path = os.path.join(models_dir, "model_" + str(i))
            unpack_model(model, path)
        else:
            logger.warning("Ignore '{}' as it is not a valid model file.".format(model))

    logger.debug("Unpacked models to compare to '{}'".format(models_dir))

    return models_dir
