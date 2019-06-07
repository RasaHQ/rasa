import asyncio
import logging
import tempfile
from typing import Text, Dict, Optional, List
import os

from rasa.core.interpreter import RegexInterpreter

from rasa.constants import DEFAULT_RESULTS_PATH
from rasa.model import get_model, get_model_subdirectories, unpack_model
from rasa.cli.utils import minimal_kwargs, print_error, print_warning

logger = logging.getLogger(__name__)


def test_compare(models: List[Text], stories: Text, output: Text):
    from rasa.core.test import compare, plot_curve
    import rasa.utils.io

    model_directory = copy_models_to_compare(models)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(compare(model_directory, stories, output))

    story_n_path = os.path.join(model_directory, "num_stories.json")
    number_of_stories = rasa.utils.io.read_json_file(story_n_path)
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

    unpacked_model = get_model(model)
    if unpacked_model is None:
        print_error(
            "Unable to test: could not find a model. Use 'rasa train' to train a "
            "Rasa model."
        )
        return

    core_path, nlu_path = get_model_subdirectories(unpacked_model)

    if not os.path.exists(core_path):
        print_error(
            "Unable to test: could not find a Core model. Use 'rasa train' to "
            "train a model."
        )

    use_e2e = kwargs["e2e"] if "e2e" in kwargs else False

    _interpreter = RegexInterpreter()
    if use_e2e:
        if os.path.exists(nlu_path):
            _interpreter = NaturalLanguageInterpreter.create(nlu_path, _endpoints.nlu)
        else:
            print_warning(
                "No NLU model found. Using default 'RegexInterpreter' for end-to-end "
                "evaluation."
            )

    _agent = Agent.load(unpacked_model, interpreter=_interpreter)

    kwargs = minimal_kwargs(kwargs, rasa.core.test, ["stories", "agent"])

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        rasa.core.test(stories, _agent, out_directory=output, **kwargs)
    )


def test_nlu(model: Optional[Text], nlu_data: Optional[Text], kwargs: Optional[Dict]):
    from rasa.nlu.test import run_evaluation

    unpacked_model = get_model(model)

    if unpacked_model is None:
        print_error(
            "Could not find any model. Use 'rasa train nlu' to train an NLU model."
        )
        return

    nlu_model = os.path.join(unpacked_model, "nlu")

    if os.path.exists(nlu_model):
        kwargs = minimal_kwargs(kwargs, run_evaluation, ["data_path", "model"])
        run_evaluation(nlu_data, nlu_model, **kwargs)
    else:
        print_error(
            "Could not find any model. Use 'rasa train nlu' to train an NLU model."
        )


def test_nlu_with_cross_validation(config: Text, nlu: Text, kwargs: Optional[Dict]):
    import rasa.nlu.config
    from rasa.nlu.test import (
        drop_intents_below_freq,
        cross_validate,
        return_results,
        return_entity_results,
    )

    kwargs = kwargs or {}
    folds = int(kwargs.get("folds", 3))
    nlu_config = rasa.nlu.config.load(config)
    data = rasa.nlu.training_data.load_data(nlu)
    data = drop_intents_below_freq(data, cutoff=folds)
    kwargs = minimal_kwargs(kwargs, cross_validate)
    results, entity_results = cross_validate(data, folds, nlu_config, **kwargs)
    logger.info("CV evaluation (n={})".format(folds))

    if any(results):
        logger.info("Intent evaluation results")
        return_results(results.train, "train")
        return_results(results.test, "test")
    if any(entity_results):
        logger.info("Entity evaluation results")
        return_entity_results(entity_results.train, "train")
        return_entity_results(entity_results.test, "test")


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
