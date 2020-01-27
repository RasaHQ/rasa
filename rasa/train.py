import asyncio
import os
import tempfile
from contextlib import ExitStack
from typing import Text, Optional, List, Union, Dict

from rasa.importers.importer import TrainingDataImporter
from rasa import model
from rasa.model import FingerprintComparisonResult
from rasa.core.domain import Domain
from rasa.utils.common import TempDirectoryPath

from rasa.cli.utils import (
    print_success,
    print_warning,
    print_error,
    bcolors,
    print_color,
)
from rasa.constants import DEFAULT_MODELS_PATH, DEFAULT_CORE_SUBDIRECTORY_NAME


def train(
    domain: Text,
    config: Text,
    training_files: Union[Text, List[Text]],
    output: Text = DEFAULT_MODELS_PATH,
    force_training: bool = False,
    fixed_model_name: Optional[Text] = None,
    persist_nlu_training_data: bool = False,
    additional_arguments: Optional[Dict] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> Optional[Text]:
    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        train_async(
            domain=domain,
            config=config,
            training_files=training_files,
            output_path=output,
            force_training=force_training,
            fixed_model_name=fixed_model_name,
            persist_nlu_training_data=persist_nlu_training_data,
            additional_arguments=additional_arguments,
        )
    )


async def train_async(
    domain: Union[Domain, Text],
    config: Text,
    training_files: Optional[Union[Text, List[Text]]],
    output_path: Text = DEFAULT_MODELS_PATH,
    force_training: bool = False,
    fixed_model_name: Optional[Text] = None,
    persist_nlu_training_data: bool = False,
    additional_arguments: Optional[Dict] = None,
) -> Optional[Text]:
    """Trains a Rasa model (Core and NLU).

    Args:
        domain: Path to the domain file.
        config: Path to the config for Core and NLU.
        training_files: Paths to the training data for Core and NLU.
        output_path: Output path.
        force_training: If `True` retrain model even if data has not changed.
        fixed_model_name: Name of model to be stored.
        persist_nlu_training_data: `True` if the NLU training data should be persisted
                                   with the model.
        additional_arguments: Additional training parameters.

    Returns:
        Path of the trained model archive.
    """

    file_importer = TrainingDataImporter.load_from_config(
        config, domain, training_files
    )
    with ExitStack() as stack:
        train_path = stack.enter_context(TempDirectoryPath(tempfile.mkdtemp()))

        domain = await file_importer.get_domain()
        if domain.is_empty():
            return await handle_domain_if_not_exists(
                file_importer, output_path, fixed_model_name
            )

        return await _train_async_internal(
            file_importer,
            train_path,
            output_path,
            force_training,
            fixed_model_name,
            persist_nlu_training_data,
            additional_arguments,
        )


async def handle_domain_if_not_exists(
    file_importer: TrainingDataImporter, output_path, fixed_model_name
):
    nlu_model_only = await _train_nlu_with_validated_data(
        file_importer, output=output_path, fixed_model_name=fixed_model_name
    )
    print_warning(
        "Core training was skipped because no valid domain file was found. Only an nlu-model was created."
        "Please specify a valid domain using '--domain' argument or check if the provided domain file exists."
    )
    return nlu_model_only


async def _train_async_internal(
    file_importer: TrainingDataImporter,
    train_path: Text,
    output_path: Text,
    force_training: bool,
    fixed_model_name: Optional[Text],
    persist_nlu_training_data: bool,
    additional_arguments: Optional[Dict],
) -> Optional[Text]:
    """Trains a Rasa model (Core and NLU). Use only from `train_async`.

    Args:
        file_importer: `TrainingDataImporter` which supplies the training data.
        train_path: Directory in which to train the model.
        output_path: Output path.
        force_training: If `True` retrain model even if data has not changed.
        persist_nlu_training_data: `True` if the NLU training data should be persisted
                                   with the model.
        fixed_model_name: Name of model to be stored.
        additional_arguments: Additional training parameters.

    Returns:
        Path of the trained model archive.
    """

    stories, nlu_data = await asyncio.gather(
        file_importer.get_stories(), file_importer.get_nlu_data()
    )

    if stories.is_empty() and nlu_data.is_empty():
        print_error(
            "No training data given. Please provide stories and NLU data in "
            "order to train a Rasa model using the '--data' argument."
        )
        return

    if stories.is_empty():
        print_warning("No stories present. Just a Rasa NLU model will be trained.")
        return await _train_nlu_with_validated_data(
            file_importer,
            output=output_path,
            fixed_model_name=fixed_model_name,
            persist_nlu_training_data=persist_nlu_training_data,
        )

    if nlu_data.is_empty():
        print_warning("No NLU data present. Just a Rasa Core model will be trained.")
        return await _train_core_with_validated_data(
            file_importer,
            output=output_path,
            fixed_model_name=fixed_model_name,
            additional_arguments=additional_arguments,
        )

    new_fingerprint = await model.model_fingerprint(file_importer)
    old_model = model.get_latest_model(output_path)
    fingerprint_comparison = FingerprintComparisonResult(force_training=force_training)
    if not force_training:
        fingerprint_comparison = model.should_retrain(
            new_fingerprint, old_model, train_path
        )

    if fingerprint_comparison.is_training_required():
        await _do_training(
            file_importer,
            output_path=output_path,
            train_path=train_path,
            fingerprint_comparison_result=fingerprint_comparison,
            fixed_model_name=fixed_model_name,
            persist_nlu_training_data=persist_nlu_training_data,
            additional_arguments=additional_arguments,
        )

        return model.package_model(
            fingerprint=new_fingerprint,
            output_directory=output_path,
            train_path=train_path,
            fixed_model_name=fixed_model_name,
        )

    print_success(
        "Nothing changed. You can use the old model stored at '{}'."
        "".format(os.path.abspath(old_model))
    )
    return old_model


async def _do_training(
    file_importer: TrainingDataImporter,
    output_path: Text,
    train_path: Text,
    fingerprint_comparison_result: Optional[FingerprintComparisonResult] = None,
    fixed_model_name: Optional[Text] = None,
    persist_nlu_training_data: bool = False,
    additional_arguments: Optional[Dict] = None,
):
    if not fingerprint_comparison_result:
        fingerprint_comparison_result = FingerprintComparisonResult()

    if fingerprint_comparison_result.should_retrain_core():
        await _train_core_with_validated_data(
            file_importer,
            output=output_path,
            train_path=train_path,
            fixed_model_name=fixed_model_name,
            additional_arguments=additional_arguments,
        )
    elif fingerprint_comparison_result.should_retrain_nlg():
        print_color(
            "Core stories/configuration did not change. "
            "Only the templates section has been changed. A new model with "
            "the updated templates will be created.",
            color=bcolors.OKBLUE,
        )
        await model.update_model_with_new_domain(file_importer, train_path)
    else:
        print_color(
            "Core stories/configuration did not change. No need to retrain Core model.",
            color=bcolors.OKBLUE,
        )

    if fingerprint_comparison_result.should_retrain_nlu():
        await _train_nlu_with_validated_data(
            file_importer,
            output=output_path,
            train_path=train_path,
            fixed_model_name=fixed_model_name,
            persist_nlu_training_data=persist_nlu_training_data,
        )
    else:
        print_color(
            "NLU data/configuration did not change. No need to retrain NLU model.",
            color=bcolors.OKBLUE,
        )


def train_core(
    domain: Union[Domain, Text],
    config: Text,
    stories: Text,
    output: Text,
    train_path: Optional[Text] = None,
    fixed_model_name: Optional[Text] = None,
    additional_arguments: Optional[Dict] = None,
) -> Optional[Text]:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        train_core_async(
            domain=domain,
            config=config,
            stories=stories,
            output=output,
            train_path=train_path,
            fixed_model_name=fixed_model_name,
            additional_arguments=additional_arguments,
        )
    )


async def train_core_async(
    domain: Union[Domain, Text],
    config: Text,
    stories: Text,
    output: Text,
    train_path: Optional[Text] = None,
    fixed_model_name: Optional[Text] = None,
    additional_arguments: Optional[Dict] = None,
) -> Optional[Text]:
    """Trains a Core model.

    Args:
        domain: Path to the domain file.
        config: Path to the config file for Core.
        stories: Path to the Core training data.
        output: Output path.
        train_path: If `None` the model will be trained in a temporary
            directory, otherwise in the provided directory.
        fixed_model_name: Name of model to be stored.
        uncompress: If `True` the model will not be compressed.
        additional_arguments: Additional training parameters.

    Returns:
        If `train_path` is given it returns the path to the model archive,
        otherwise the path to the directory with the trained model files.

    """

    file_importer = TrainingDataImporter.load_core_importer_from_config(
        config, domain, [stories]
    )
    domain = await file_importer.get_domain()
    if domain.is_empty():
        print_error(
            "Core training was skipped because no valid domain file was found. "
            "Please specify a valid domain using '--domain' argument or check if the provided domain file exists."
        )
        return None

    if not await file_importer.get_stories():
        print_error(
            "No stories given. Please provide stories in order to "
            "train a Rasa Core model using the '--stories' argument."
        )
        return

    return await _train_core_with_validated_data(
        file_importer,
        output=output,
        train_path=train_path,
        fixed_model_name=fixed_model_name,
        additional_arguments=additional_arguments,
    )


async def _train_core_with_validated_data(
    file_importer: TrainingDataImporter,
    output: Text,
    train_path: Optional[Text] = None,
    fixed_model_name: Optional[Text] = None,
    additional_arguments: Optional[Dict] = None,
) -> Optional[Text]:
    """Train Core with validated training and config data."""

    import rasa.core.train

    with ExitStack() as stack:
        if train_path:
            # If the train path was provided, do nothing on exit.
            _train_path = train_path
        else:
            # Otherwise, create a temp train path and clean it up on exit.
            _train_path = stack.enter_context(TempDirectoryPath(tempfile.mkdtemp()))

        # normal (not compare) training
        print_color("Training Core model...", color=bcolors.OKBLUE)
        domain, config = await asyncio.gather(
            file_importer.get_domain(), file_importer.get_config()
        )
        await rasa.core.train(
            domain_file=domain,
            training_resource=file_importer,
            output_path=os.path.join(_train_path, DEFAULT_CORE_SUBDIRECTORY_NAME),
            policy_config=config,
            additional_arguments=additional_arguments,
        )
        print_color("Core model training completed.", color=bcolors.OKBLUE)

        if train_path is None:
            # Only Core was trained.
            new_fingerprint = await model.model_fingerprint(file_importer)
            return model.package_model(
                fingerprint=new_fingerprint,
                output_directory=output,
                train_path=_train_path,
                fixed_model_name=fixed_model_name,
                model_prefix="core-",
            )

        return _train_path


def train_nlu(
    config: Text,
    nlu_data: Text,
    output: Text,
    train_path: Optional[Text] = None,
    fixed_model_name: Optional[Text] = None,
    persist_nlu_training_data: bool = False,
) -> Optional[Text]:
    """Trains an NLU model.

    Args:
        config: Path to the config file for NLU.
        nlu_data: Path to the NLU training data.
        output: Output path.
        train_path: If `None` the model will be trained in a temporary
            directory, otherwise in the provided directory.
        fixed_model_name: Name of the model to be stored.
        persist_nlu_training_data: `True` if the NLU training data should be persisted
                                   with the model.


    Returns:
        If `train_path` is given it returns the path to the model archive,
        otherwise the path to the directory with the trained model files.

    """

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        _train_nlu_async(
            config,
            nlu_data,
            output,
            train_path,
            fixed_model_name,
            persist_nlu_training_data,
        )
    )


async def _train_nlu_async(
    config: Text,
    nlu_data: Text,
    output: Text,
    train_path: Optional[Text] = None,
    fixed_model_name: Optional[Text] = None,
    persist_nlu_training_data: bool = False,
):
    # training NLU only hence the training files still have to be selected
    file_importer = TrainingDataImporter.load_nlu_importer_from_config(
        config, training_data_paths=[nlu_data]
    )

    training_datas = await file_importer.get_nlu_data()
    if training_datas.is_empty():
        print_error(
            "No NLU data given. Please provide NLU data in order to train "
            "a Rasa NLU model using the '--nlu' argument."
        )
        return

    return await _train_nlu_with_validated_data(
        file_importer,
        output=output,
        train_path=train_path,
        fixed_model_name=fixed_model_name,
        persist_nlu_training_data=persist_nlu_training_data,
    )


async def _train_nlu_with_validated_data(
    file_importer: TrainingDataImporter,
    output: Text,
    train_path: Optional[Text] = None,
    fixed_model_name: Optional[Text] = None,
    persist_nlu_training_data: bool = False,
) -> Optional[Text]:
    """Train NLU with validated training and config data."""

    import rasa.nlu.train

    with ExitStack() as stack:
        if train_path:
            # If the train path was provided, do nothing on exit.
            _train_path = train_path
        else:
            # Otherwise, create a temp train path and clean it up on exit.
            _train_path = stack.enter_context(TempDirectoryPath(tempfile.mkdtemp()))
        config = await file_importer.get_config()
        print_color("Training NLU model...", color=bcolors.OKBLUE)
        _, nlu_model, _ = await rasa.nlu.train(
            config,
            file_importer,
            _train_path,
            fixed_model_name="nlu",
            persist_nlu_training_data=persist_nlu_training_data,
        )
        print_color("NLU model training completed.", color=bcolors.OKBLUE)

        if train_path is None:
            # Only NLU was trained
            new_fingerprint = await model.model_fingerprint(file_importer)

            return model.package_model(
                fingerprint=new_fingerprint,
                output_directory=output,
                train_path=_train_path,
                fixed_model_name=fixed_model_name,
                model_prefix="nlu-",
            )

        return _train_path
