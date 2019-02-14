import argparse
import logging
import os

from rasa.cli.default_arguments import add_model_param, add_stories_param
from rasa.cli.utils import validate
from rasa.cli.constants import (DEFAULT_ENDPOINTS_PATH,
                                DEFAULT_CONFIG_PATH, DEFAULT_NLU_DATA_PATH)
from rasa.model import DEFAULT_MODELS_PATH, get_latest_model, get_model

logger = logging.getLogger(__name__)


def add_subparser(subparsers, parents):
    test_parser = subparsers.add_parser(
        "test",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Test a trained model")

    test_subparsers = test_parser.add_subparsers()
    test_core_parser = test_subparsers.add_parser(
        "core",
        conflict_handler='resolve',
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Test Rasa Core")

    test_nlu_parser = test_subparsers.add_parser(
        "nlu",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Test Rasa NLU")

    for p in [test_parser, test_core_parser]:
        core_arguments = p.add_argument_group("Core test arguments")
        _add_core_arguments(core_arguments)
    _add_core_subparser_arguments(test_core_parser)

    for p in [test_parser, test_nlu_parser]:
        nlu_arguments = p.add_argument_group("NLU test arguments")
        _add_nlu_arguments(nlu_arguments)
    _add_nlu_subparser_arguments(test_nlu_parser)

    test_core_parser.set_defaults(func=test_core)
    test_nlu_parser.set_defaults(func=test_nlu)
    test_parser.set_defaults(func=test)


def _add_core_arguments(parser):
    from rasa_core.cli.evaluation import add_evaluation_arguments

    add_evaluation_arguments(parser)
    add_model_param(parser, "Core")
    add_stories_param(parser, "test")

    parser.add_argument(
        '--url',
        type=str,
        help="If supplied, downloads a story file from a URL and "
             "trains on it. Fetches the data by sending a GET request "
             "to the supplied URL.")


def _add_core_subparser_arguments(parser):
    default_path=get_latest_model(DEFAULT_MODELS_PATH)
    parser.add_argument(
        '-m', '--model',
        type=str,
        default=default_path,
        help="Path to a pre-trained model. If it is a directory all models "
             "in this directory will be compared.")


def _add_nlu_arguments(parser):
    parser.add_argument('-u', '--nlu',
                        type=str,
                        default="data/nlu",
                        help="file containing training/evaluation data")

    parser.add_argument('-c', '--config',
                        type=str,
                        default="config.yml",
                        help="model configuration file (crossvalidation only)")

    parser.add_argument('-f', '--folds', required=False, default=10,
                        help="number of CV folds (crossvalidation only)")

    parser.add_argument('--report', required=False, nargs='?',
                        const="reports", default=False,
                        help="output path to save the intent/entity"
                             "metrics report")

    parser.add_argument('--successes', required=False, nargs='?',
                        const="successes.json", default=False,
                        help="output path to save successful predictions")

    parser.add_argument('--errors', required=False, default="errors.json",
                        help="output path to save model errors")

    parser.add_argument('--histogram', required=False, default="hist.png",
                        help="output path for the confidence histogram")

    parser.add_argument('--confmat', required=False, default="confmat.png",
                        help="output path for the confusion matrix plot")


def _add_nlu_subparser_arguments(parser):
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help="Path to a pre-trained model. If none is given it will "
             "perform crossvalidation.")


def test_core(args, model_path=None):
    import rasa_core.evaluate
    from rasa_nlu import utils as nlu_utils
    from rasa_core.utils import AvailableEndpoints
    from rasa_core.interpreter import NaturalLanguageInterpreter
    from rasa_core.agent import Agent

    validate(args, [("model", DEFAULT_MODELS_PATH),
                    ("endpoints", DEFAULT_ENDPOINTS_PATH, True),
                    ("config", DEFAULT_CONFIG_PATH)])

    _endpoints = AvailableEndpoints.read_endpoints(
        args.endpoints)

    if args.output:
        nlu_utils.create_dir(args.output)

    if os.path.isfile(args.model):
        model_path = get_model(args.model)

    if model_path:
        # Single model: Normal evaluation
        model_path = get_model(args.model)
        core_path = os.path.join(model_path, "core")
        nlu_path = os.path.join(model_path, "nlu")

        _interpreter = NaturalLanguageInterpreter.create(
            nlu_path,
            _endpoints.nlu)

        _agent = Agent.load(core_path,
                            interpreter=_interpreter)

        stories = rasa_core.cli.stories_from_cli_args(args)

        rasa_core.evaluate.run_story_evaluation(stories,
                                                _agent,
                                                args.max_stories,
                                                args.output,
                                                args.fail_on_prediction_errors,
                                                args.e2e)
    else:
        rasa_core.evaluate.run_comparison_evaluation(args.model,
                                                     args.stories, args.output)

        story_n_path = os.path.join(args.core, 'num_stories.json')

        number_of_stories = rasa_core.utils.read_json_file(story_n_path)
        rasa_core.evaluate.plot_curve(args.output, number_of_stories)


def test_nlu(args, model_path=None):
    import rasa_nlu

    validate(args, [("model", DEFAULT_MODELS_PATH),
                    ("nlu", DEFAULT_NLU_DATA_PATH)])

    model_path = model_path or args.model
    if model_path:
        unpacked_model = get_model(args.model)

        nlu_model = os.path.join(unpacked_model, "nlu")

        rasa_nlu.evaluate.run_evaluation(args.nlu,
                                         nlu_model,
                                         args.report,
                                         args.successes,
                                         args.errors,
                                         args.confmat,
                                         args.histogram)
    else:
        print("No model specified. Model will be trained using cross "
              "validation.")

        nlu_config = rasa_nlu.config.load(args.config)
        data = rasa_nlu.training_data.load_data(args.nlu)
        data = rasa_nlu.evaluate.drop_intents_below_freq(data, cutoff=5)
        results, entity_results = rasa_nlu.evaluate.run_cv_evaluation(
            data, int(args.folds), nlu_config)
        logger.info("CV evaluation (n={})".format(args.folds))

        if any(results):
            logger.info("Intent evaluation results")
            rasa_nlu.evaluate.return_results(results.train, "train")
            rasa_nlu.evaluate.return_results(results.test, "test")
        if any(entity_results):
            logger.info("Entity evaluation results")
            rasa_nlu.evaluate.return_entity_results(entity_results.train,
                                                    "train")
            rasa_nlu.evaluate.return_entity_results(entity_results.test,
                                                    "test")


def test(args):
    validate(args, [("model", DEFAULT_MODELS_PATH)])
    model_path = get_model(args.model)

    test_core(args, model_path)
    test_nlu(args, model_path)
