import asyncio
import logging
import tempfile
from typing import Text, Dict, Optional, List
import os

from rasa.core.interpreter import RegexInterpreter

from rasa.constants import DEFAULT_RESULTS_PATH
from rasa.model import get_model, get_model_subdirectories, unpack_model
from rasa.cli.utils import (
    minimal_kwargs,
    print_error,
    print_warning,
    print_info,
    print_success,
)

logger = logging.getLogger(__name__)


def test_compare_core(models: List[Text], stories: Text, output: Text):
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


def test_compare_nlu(
    configs: List[Text],
    nlu: Text,
    output: Text,
    runs: int,
    exclusion_percentages: List[int],
):
    """Trains multiple models, compares them and saves the results."""
    from rasa.nlu.test import drop_intents_below_freq, run_evaluation
    from rasa.nlu.training_data import load_data
    from rasa.nlu.utils import write_to_file, write_json_to_file
    from rasa.utils.io import create_path
    from rasa.train import train_nlu
    from rasa.core.test import plot_curve
    from rasa.model import get_model

    data = load_data(nlu)
    data = drop_intents_below_freq(data, cutoff=5)

    create_path(output)

    bases = [os.path.basename(nlu_config) for nlu_config in configs]
    model_names = [os.path.splitext(base)[0] for base in bases]
    micros = dict((model_name, [[] for _ in range(runs)]) for model_name in model_names)

    for run in range(runs):

        print_info("Beginning comparison run {}/{}".format(run + 1, runs))
        train, test = data.train_test_split()

        run_path = os.path.join(output, "run_{}".format(run + 1))
        create_path(run_path)

        test_path = os.path.join(run_path, "test.md")
        create_path(test_path)

        write_to_file(test_path, test.as_markdown())
        intent_examples_present = []

        for percentage in exclusion_percentages:
            percent_string = "{}%_exclusion".format(percentage)

            _, train = train.train_test_split(percentage / 100)
            intent_examples_present.append(len(train.training_examples))

            out_path = os.path.join(run_path, percent_string)
            train_split_path = os.path.join(out_path, "train.md")
            create_path(train_split_path)

            write_to_file(train_split_path, train.as_markdown())

            for nlu_config, model_name in zip(configs, model_names):

                print_success(
                    "Evaluating config '{}' with {}".format(model_name, percent_string)
                )

                model_path = train_nlu(
                    nlu_config, train_split_path, out_path, fixed_model_name=model_name
                )

                model_path = os.path.join(get_model(model_path), "nlu")

                report_path = os.path.join(out_path, "{}_report".format(model_name))
                errors_path = os.path.join(report_path, "errors.json")
                result = run_evaluation(
                    test_path, model_path, report=report_path, errors=errors_path
                )

                f1 = result["intent_evaluation"]["f1_score"]
                micros[model_name][run].append(f1)

    f1_path = os.path.join(output, "results.json")
    write_json_to_file(f1_path, micros)

    plot_curve(output, intent_examples_present, mode="nlu")


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
