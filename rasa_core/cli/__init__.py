import rasa_core.cli.arguments
from rasa_core import utils


def stories_from_cli_args(cmdline_arguments):
    if cmdline_arguments.url:
        return utils.download_file_from_url(cmdline_arguments.url)
    else:
        return cmdline_arguments.stories
