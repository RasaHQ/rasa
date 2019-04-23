from rasa.core.cli import arguments


def add_compare_args(parser):
    parser.add_argument(
        "--percentages",
        nargs="*",
        type=int,
        default=[0, 5, 25, 50, 70, 90, 95],
        help="Range of exclusion percentages",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of runs for experiments"
    )

    arguments.add_output_arg(
        parser, help_text="directory to persist the trained model in", required=True
    )
    arguments.add_config_arg(parser, nargs="*")
    arguments.add_model_and_story_group(parser, allow_pretrained_model=False)
    arguments.add_domain_arg(parser, required=True)


def add_interactive_args(parser):
    parser.add_argument("-u", "--nlu", type=str, default=None, help="trained nlu model")
    parser.add_argument(
        "--endpoints",
        default=None,
        help="Configuration file for the connectors as a yml file",
    )
    parser.add_argument(
        "--skip_visualization",
        default=False,
        action="store_true",
        help="disables plotting the visualization during interactive learning",
    )
    parser.add_argument(
        "--finetune",
        default=False,
        action="store_true",
        help="retrain the model immediately based on feedback.",
    )
    parser.add_argument(
        "--nlu_data",
        default=None,
        help="Location where the nlu training data should be saved.",
    )

    arguments.add_output_arg(
        parser, help_text="directory to persist the trained model in", required=False
    )
    arguments.add_config_arg(parser, nargs=1)
    arguments.add_model_and_story_group(parser, allow_pretrained_model=True)
    arguments.add_domain_arg(parser, required=False)


def add_train_args(parser):
    arguments.add_config_arg(parser, nargs=1)
    arguments.add_output_arg(
        parser, help_text="directory to persist the trained model in", required=True
    )
    arguments.add_model_and_story_group(parser, allow_pretrained_model=False)
    arguments.add_domain_arg(parser, required=True)


def add_general_args(parser):
    parser.add_argument(
        "--augmentation",
        type=int,
        default=50,
        help="how much data augmentation to use during training",
    )
    parser.add_argument(
        "--dump_stories",
        default=False,
        action="store_true",
        help="If enabled, save flattened stories to a file",
    )
    parser.add_argument(
        "--debug_plots",
        default=False,
        action="store_true",
        help="If enabled, will create plots showing checkpoints "
        "and their connections between story blocks in a  "
        "file called `story_blocks_connections.html`.",
    )

    arguments.add_logging_option_arguments(parser)


async def stories_from_cli_args(cmdline_arguments):
    from rasa.core import utils

    if cmdline_arguments.url:
        return await utils.download_file_from_url(cmdline_arguments.url)
    else:
        return cmdline_arguments.stories
