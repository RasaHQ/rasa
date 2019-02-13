from rasa.cli.utils import check_path_exists
from rasa.model import DEFAULT_MODELS_PATH


def add_model_param(parser, model_name="Rasa"):
    parser.add_argument("-m", "--model",
                        type=lambda v: check_path_exists(v, "--model",
                                                         DEFAULT_MODELS_PATH),
                        default=DEFAULT_MODELS_PATH,
                        help="Path to a trained {} model. If a directory "
                             "is specified, it will use the latest model "
                             "in this directory.".format(model_name))


def add_stories_param(parser, stories_name="training"):
    parser.add_argument(
        "-s", "--stories",
        type=lambda v: check_path_exists(v, "--stories", "data/core"),
        default="data/core",
        help="File or folder containing {} stories.".format(stories_name))


def add_domain_param(parser):
    parser.add_argument("-d", "--domain",
                        type=lambda v: check_path_exists(v, "--domain",
                                                         "domain.yml"),
                        default="domain.yml",
                        help="Domain specification (yml file)")


def add_config_param(parser):
    parser.add_argument(
        "-c", "--config",
        type=lambda v: check_path_exists(v, "--config", "config.yml"),
        default="config.yml",
        help="The policy and NLU pipeline configuration of your bot.")
