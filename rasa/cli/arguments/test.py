import argparse
from typing import Union

from rasa.shared.constants import DEFAULT_MODELS_PATH, DEFAULT_RESULTS_PATH

from rasa.cli.arguments.default_arguments import (
    add_stories_param,
    add_model_param,
    add_nlu_data_param,
    add_endpoint_param,
    add_out_param,
)
from rasa.model import get_latest_model


def set_test_arguments(parser: argparse.ArgumentParser) -> None:
    add_model_param(parser, add_positional_arg=False)

    core_arguments = parser.add_argument_group("Core Test Arguments")
    add_test_core_argument_group(core_arguments)

    nlu_arguments = parser.add_argument_group("NLU Test Arguments")
    add_test_nlu_argument_group(nlu_arguments)

    add_no_plot_param(parser)
    add_errors_success_params(parser)
    add_out_param(
        parser,
        default=DEFAULT_RESULTS_PATH,
        help_text="Output path for any files created during the evaluation.",
    )


def set_test_core_arguments(parser: argparse.ArgumentParser) -> None:
    add_test_core_model_param(parser)
    add_test_core_argument_group(parser, include_e2e_argument=True)


def set_test_nlu_arguments(parser: argparse.ArgumentParser) -> None:
    add_model_param(parser, add_positional_arg=False)
    add_test_nlu_argument_group(parser)


def add_test_core_argument_group(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
    include_e2e_argument: bool = False,
) -> None:
    add_stories_param(parser, "test")
    parser.add_argument(
        "--max-stories", type=int, help="Maximum number of stories to test on."
    )
    add_out_param(
        parser,
        default=DEFAULT_RESULTS_PATH,
        help_text="Output path for any files created during the evaluation.",
    )
    if include_e2e_argument:
        parser.add_argument(
            "--e2e",
            "--end-to-end",
            action="store_true",
            help="Run an end-to-end evaluation for combined action and "
            "intent prediction. Requires a story file in end-to-end "
            "format.",
        )
    add_endpoint_param(
        parser, help_text="Configuration file for the connectors as a yml file."
    )
    parser.add_argument(
        "--fail-on-prediction-errors",
        action="store_true",
        help="If a prediction error is encountered, an exception "
        "is thrown. This can be used to validate stories during "
        "tests, e.g. on travis.",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="If supplied, downloads a story file from a URL and "
        "trains on it. Fetches the data by sending a GET request "
        "to the supplied URL.",
    )
    parser.add_argument(
        "--evaluate-model-directory",
        default=False,
        action="store_true",
        help="Should be set to evaluate models trained via "
        "'rasa train core --config <config-1> <config-2>'. "
        "All models in the provided directory are evaluated "
        "and compared against each other.",
    )
    add_no_plot_param(parser)
    add_errors_success_params(parser)


def add_test_nlu_argument_group(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
) -> None:
    add_nlu_data_param(parser, help_text="File or folder containing your NLU data.")

    add_out_param(
        parser,
        default=DEFAULT_RESULTS_PATH,
        help_text="Output path for any files created during the evaluation.",
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="+",
        default=None,
        help="Model configuration file. If a single file is passed and cross "
        "validation mode is chosen, cross-validation is performed, if "
        "multiple configs or a folder of configs are passed, models "
        "will be trained and compared directly.",
    )

    cross_validation_arguments = parser.add_argument_group("Cross Validation")
    cross_validation_arguments.add_argument(
        "--cross-validation",
        action="store_true",
        default=False,
        help="Switch on cross validation mode. Any provided model will be ignored.",
    )
    cross_validation_arguments.add_argument(
        "-f",
        "--folds",
        required=False,
        default=5,
        help="Number of cross validation folds (cross validation only).",
    )
    comparison_arguments = parser.add_argument_group("Comparison Mode")
    comparison_arguments.add_argument(
        "-r",
        "--runs",
        required=False,
        default=3,
        type=int,
        help="Number of comparison runs to make.",
    )
    comparison_arguments.add_argument(
        "-p",
        "--percentages",
        required=False,
        nargs="+",
        type=int,
        default=[0, 25, 50, 75],
        help="Percentages of training data to exclude during comparison.",
    )

    add_no_plot_param(parser)
    add_errors_success_params(parser)


def add_test_core_model_param(parser: argparse.ArgumentParser) -> None:
    default_path = get_latest_model(DEFAULT_MODELS_PATH)
    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        default=[default_path],
        help="Path to a pre-trained model. If it is a 'tar.gz' file that model file "
        "will be used. If it is a directory, the latest model in that directory "
        "will be used (exception: '--evaluate-model-directory' flag is set). "
        "If multiple 'tar.gz' files are provided, all those models will be compared.",
    )


def add_no_plot_param(
    parser: argparse.ArgumentParser, default: bool = False, required: bool = False
) -> None:
    parser.add_argument(
        "--no-plot",
        dest="disable_plotting",
        action="store_true",
        default=default,
        help="Don't render evaluation plots.",
        required=required,
    )


def add_errors_success_params(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--successes",
        action="store_true",
        default=False,
        help="If set successful predictions will be written to a file.",
    )
    parser.add_argument(
        "--no-errors",
        action="store_true",
        default=False,
        help="If set incorrect predictions will NOT be written to a file.",
    )
    parser.add_argument(
        "--no-warnings",
        action="store_true",
        default=False,
        help="If set prediction warnings will NOT be written to a file.",
    )
