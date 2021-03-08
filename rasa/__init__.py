import logging
import typing

from rasa import version
import rasa.shared.constants as __shared_constants

if typing.TYPE_CHECKING:
    from typing import Any, Text, Dict, Union, List, Optional
    from rasa.train import TrainingResult
    import asyncio

# define the version before the other imports since these need it
__version__ = version.__version__


logging.getLogger(__name__).addHandler(logging.NullHandler())


def run(
    model: "Text",
    endpoints: "Text",
    connector: "Text" = None,
    credentials: "Text" = None,
    **kwargs: "Dict[Text, Any]",
):
    """Runs a Rasa model.

    Args:
        model: Path to model archive.
        endpoints: Path to endpoints file.
        connector: Connector which should be use (overwrites `credentials`
        field).
        credentials: Path to channel credentials file.
        **kwargs: Additional arguments which are passed to
        `rasa.core.run.serve_application`.

    """
    import rasa.core.run
    import rasa.nlu.run
    from rasa.core.utils import AvailableEndpoints
    from rasa.shared.utils.cli import print_warning
    import rasa.shared.utils.common
    from rasa.shared.constants import DOCS_BASE_URL

    _endpoints = AvailableEndpoints.read_endpoints(endpoints)

    if not connector and not credentials:
        connector = "rest"

        print_warning(
            f"No chat connector configured, falling back to the "
            f"REST input channel. To connect your bot to another channel, "
            f"read the docs here: {DOCS_BASE_URL}/messaging-and-voice-channels"
        )

    kwargs = rasa.shared.utils.common.minimal_kwargs(
        kwargs, rasa.core.run.serve_application
    )
    rasa.core.run.serve_application(
        model,
        channel=connector,
        credentials=credentials,
        endpoints=_endpoints,
        **kwargs,
    )


def train(
    domain: "Text",
    config: "Text",
    training_files: "Union[Text, List[Text]]",
    output: "Text" = __shared_constants.DEFAULT_MODELS_PATH,
    dry_run: bool = False,
    force_training: bool = False,
    fixed_model_name: "Optional[Text]" = None,
    persist_nlu_training_data: bool = False,
    core_additional_arguments: "Optional[Dict]" = None,
    nlu_additional_arguments: "Optional[Dict]" = None,
    loop: "Optional[asyncio.AbstractEventLoop]" = None,
    model_to_finetune: "Optional[Text]" = None,
    finetuning_epoch_fraction: float = 1.0,
) -> "TrainingResult":
    """Runs Rasa Core and NLU training in `async` loop.

    Args:
        domain: Path to the domain file.
        config: Path to the config for Core and NLU.
        training_files: Paths to the training data for Core and NLU.
        output: Output path.
        dry_run: If `True` then no training will be done, and the information about
            whether the training needs to be done will be printed.
        force_training: If `True` retrain model even if data has not changed.
        fixed_model_name: Name of model to be stored.
        persist_nlu_training_data: `True` if the NLU training data should be persisted
            with the model.
        core_additional_arguments: Additional training parameters for core training.
        nlu_additional_arguments: Additional training parameters forwarded to training
            method of each NLU component.
        loop: Optional EventLoop for running coroutines.
        model_to_finetune: Optional path to a model which should be finetuned or
            a directory in case the latest trained model should be used.
        finetuning_epoch_fraction: The fraction currently specified training epochs
            in the model configuration which should be used for finetuning.

    Returns:
        An instance of `TrainingResult`.
    """
    from rasa.train import train_async
    import rasa.utils.common

    return rasa.utils.common.run_in_loop(
        train_async(
            domain=domain,
            config=config,
            training_files=training_files,
            output=output,
            dry_run=dry_run,
            force_training=force_training,
            fixed_model_name=fixed_model_name,
            persist_nlu_training_data=persist_nlu_training_data,
            core_additional_arguments=core_additional_arguments,
            nlu_additional_arguments=nlu_additional_arguments,
            model_to_finetune=model_to_finetune,
            finetuning_epoch_fraction=finetuning_epoch_fraction,
        ),
        loop,
    )


def test(
    model: "Text",
    stories: "Text",
    nlu_data: "Text",
    output: "Text" = __shared_constants.DEFAULT_RESULTS_PATH,
    additional_arguments: "Optional[Dict]" = None,
) -> None:
    from rasa.test import test_core
    import rasa.utils.common
    from rasa.test import test_nlu

    if additional_arguments is None:
        additional_arguments = {}

    test_core(model, stories, output, additional_arguments)

    rasa.utils.common.run_in_loop(
        test_nlu(model, nlu_data, output, additional_arguments)
    )
