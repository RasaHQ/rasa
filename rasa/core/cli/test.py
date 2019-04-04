from rasa.core.cli import arguments


def add_evaluation_arguments(parser):
    parser.add_argument(
        "-m", "--max_stories", type=int, help="maximum number of stories to test on"
    )
    parser.add_argument(
        "-u",
        "--nlu",
        type=str,
        help="nlu model to run with the server. None for regex interpreter",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results",
        help="output path for the any files created from the evaluation",
    )
    parser.add_argument(
        "--e2e",
        "--end-to-end",
        action="store_true",
        help="Run an end-to-end evaluation for combined action and "
        "intent prediction. Requires a story file in end-to-end "
        "format.",
    )
    parser.add_argument(
        "--endpoints",
        default=None,
        help="Configuration file for the connectors as a yml file",
    )
    parser.add_argument(
        "--fail_on_prediction_errors",
        action="store_true",
        help="If a prediction error is encountered, an exception "
        "is thrown. This can be used to validate stories during "
        "tests, e.g. on travis.",
    )

    arguments.add_core_model_arg(parser)
