import argparse
import uuid

from rasa.cli.arguments.default_arguments import (
    add_domain_param,
    add_stories_param,
    add_model_param,
    add_endpoint_param,
)
from rasa.cli.arguments.train import (
    add_force_param,
    add_data_param,
    add_config_param,
    add_out_param,
    add_debug_plots_param,
    add_augmentation_param,
    add_persist_nlu_data_param,
)
from rasa.cli.arguments.run import add_port_argument


def set_interactive_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Save story files in e2e format. In this format user messages "
        "will be included in the stories.",
    )
    add_port_argument(parser)

    add_model_param(parser, default=None)
    add_data_param(parser)

    _add_common_params(parser)
    train_arguments = _add_training_arguments(parser)

    add_force_param(train_arguments)
    add_persist_nlu_data_param(train_arguments)


def set_interactive_core_arguments(parser: argparse.ArgumentParser) -> None:
    add_model_param(parser, model_name="Rasa Core", default=None)
    add_stories_param(parser)

    _add_common_params(parser)
    _add_training_arguments(parser)
    add_port_argument(parser)


def _add_common_params(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-visualization",
        default=False,
        action="store_true",
        help="Disable plotting the visualization during interactive learning.",
    )

    parser.add_argument(
        "--conversation-id",
        default=uuid.uuid4().hex,
        help="Specify the id of the conversation the messages are in. Defaults to a "
        "UUID that will be randomly generated.",
    )

    add_endpoint_param(
        parser,
        help_text="Configuration file for the model server and the connectors as a yml file.",
    )


# noinspection PyProtectedMember
def _add_training_arguments(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    train_arguments = parser.add_argument_group("Train Arguments")
    add_config_param(train_arguments)
    add_domain_param(train_arguments)
    add_out_param(
        train_arguments, help_text="Directory where your models should be stored."
    )
    add_augmentation_param(train_arguments)
    add_debug_plots_param(train_arguments)

    return train_arguments
