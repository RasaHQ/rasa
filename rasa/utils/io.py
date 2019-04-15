import asyncio
import io
import logging
import warnings
from asyncio import AbstractEventLoop
from typing import Text, Any, Dict
import ruamel.yaml as yaml


def configure_colored_logging(loglevel):
    import coloredlogs

    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
    field_styles["asctime"] = {}
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
    level_styles["debug"] = {}
    coloredlogs.install(
        level=loglevel,
        use_chroot=False,
        fmt="%(asctime)s %(levelname)-8s %(name)s  - %(message)s",
        level_styles=level_styles,
        field_styles=field_styles,
    )


def enable_async_loop_debugging(event_loop: AbstractEventLoop) -> AbstractEventLoop:
    logging.info(
        "Enabling coroutine debugging. Loop id {}.".format(id(asyncio.get_event_loop()))
    )

    # Enable debugging
    event_loop.set_debug(True)

    # Make the threshold for "slow" tasks very very small for
    # illustration. The default is 0.1 (= 100 milliseconds).
    event_loop.slow_callback_duration = 0.001

    # Report all mistakes managing asynchronous resources.
    warnings.simplefilter("always", ResourceWarning)
    return event_loop


def fix_yaml_loader() -> None:
    """Ensure that any string read by yaml is represented as unicode."""

    def construct_yaml_str(self, node):
        # Override the default string handling function
        # to always return unicode objects
        return self.construct_scalar(node)

    yaml.Loader.add_constructor("tag:yaml.org,2002:str", construct_yaml_str)
    yaml.SafeLoader.add_constructor("tag:yaml.org,2002:str", construct_yaml_str)


def replace_environment_variables():
    """Enable yaml loader to process the environment variables in the yaml."""
    import re
    import os

    # eg. ${USER_NAME}, ${PASSWORD}
    env_var_pattern = re.compile(r"^(.*)\$\{(.*)\}(.*)$")
    yaml.add_implicit_resolver("!env_var", env_var_pattern)

    def env_var_constructor(loader, node):
        """Process environment variables found in the YAML."""
        value = loader.construct_scalar(node)
        expanded_vars = os.path.expandvars(value)
        if "$" in expanded_vars:
            not_expanded = [w for w in expanded_vars.split() if "$" in w]
            raise ValueError(
                "Error when trying to expand the environment variables"
                " in '{}'. Please make sure to also set these environment"
                " variables: '{}'.".format(value, not_expanded)
            )
        return expanded_vars

    yaml.SafeConstructor.add_constructor("!env_var", env_var_constructor)


def read_yaml(content: Text) -> Dict[Text, Any]:
    """Parses yaml from a text.

     Args:
        content: A text containing yaml content.
    """
    fix_yaml_loader()
    replace_environment_variables()

    yaml_parser = yaml.YAML(typ="safe")
    yaml_parser.version = "1.2"
    yaml_parser.unicode_supplementary = True

    # noinspection PyUnresolvedReferences
    try:
        return yaml_parser.load(content) or {}
    except yaml.scanner.ScannerError as _:
        # A `ruamel.yaml.scanner.ScannerError` might happen due to escaped
        # unicode sequences that form surrogate pairs. Try converting the input
        # to a parsable format based on
        # https://stackoverflow.com/a/52187065/3429596.
        content = (
            content.encode("utf-8")
            .decode("raw_unicode_escape")
            .encode("utf-16", "surrogatepass")
            .decode("utf-16")
        )
        return yaml_parser.load(content) or {}


def read_file(filename: Text, encoding: Text = "utf-8") -> Any:
    """Read text from a file."""
    with io.open(filename, encoding=encoding) as f:
        return f.read()


def read_yaml_file(filename: Text) -> Dict[Text, Any]:
    """Parses a yaml file.

     Args:
        filename: The path to the file which should be read.
    """
    return read_yaml(read_file(filename, "utf-8"))
