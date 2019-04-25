import asyncio
import os
import tempfile
import typing
from typing import Text, Optional, List, Union, Dict

from rasa import model, data
from rasa.cli.utils import create_output_path, print_success
from rasa.constants import DEFAULT_MODELS_PATH

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Interpreter


def train(
    domain: Text,
    config: Text,
    training_files: Union[Text, List[Text]],
    output: Text = DEFAULT_MODELS_PATH,
    force_training: bool = False,
    kwargs: Optional[Dict] = None,
) -> Optional[Text]:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        train_async(domain, config, training_files, output, force_training, kwargs)
    )


async def train_async(
    domain: Text,
    config: Text,
    training_files: Union[Text, List[Text]],
    output: Text = DEFAULT_MODELS_PATH,
    force_training: bool = False,
    kwargs: Optional[Dict] = None,
) -> Optional[Text]:
    """Trains a Rasa model (Core and NLU).

    Args:
        domain: Path to the domain file.
        config: Path to the config for Core and NLU.
        training_files: Paths to the training data for Core and NLU.
        output: Output path.
        force_training: If `True` retrain model even if data has not changed.
        kwargs: Additional training parameters.

    Returns:
        Path of the trained model archive.
    """

    train_path = tempfile.mkdtemp()
    old_model = model.get_latest_model(output)
    retrain_core = True
    retrain_nlu = True

    story_directory, nlu_data_directory = data.get_core_nlu_directories(training_files)
    new_fingerprint = model.model_fingerprint(
        config, domain, nlu_data_directory, story_directory
    )

    if not force_training and old_model:
        unpacked = model.unpack_model(old_model)
        old_core, old_nlu = model.get_model_subdirectories(unpacked)
        last_fingerprint = model.fingerprint_from_path(unpacked)

        if not model.core_fingerprint_changed(last_fingerprint, new_fingerprint):
            target_path = os.path.join(train_path, "core")
            retrain_core = not model.merge_model(old_core, target_path)

        if not model.nlu_fingerprint_changed(last_fingerprint, new_fingerprint):
            target_path = os.path.join(train_path, "nlu")
            retrain_nlu = not model.merge_model(old_nlu, target_path)

    if force_training or retrain_core:
        await train_core_async(
            domain, config, story_directory, output, train_path, kwargs
        )
    else:
        print (
            "Dialogue data / configuration did not change. "
            "No need to retrain dialogue model."
        )

    if force_training or retrain_nlu:
        train_nlu(config, nlu_data_directory, output, train_path)
    else:
        print ("NLU data / configuration did not change. No need to retrain NLU model.")

    if retrain_core or retrain_nlu:
        output = create_output_path(output)
        model.create_package_rasa(train_path, output, new_fingerprint)

        print ("Train path: '{}'.".format(train_path))

        print_success("Your bot is trained and ready to take for a spin!")

        return output
    else:
        print (
            "Nothing changed. You can use the old model stored at {}"
            "".format(os.path.abspath(old_model))
        )

        return old_model


def train_core(
    domain: Text,
    config: Text,
    stories: Text,
    output: Text,
    train_path: Optional[Text],
    kwargs: Optional[Dict],
) -> Optional[Text]:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        train_core_async(domain, config, stories, output, train_path, kwargs)
    )


async def train_core_async(
    domain: Text,
    config: Text,
    stories: Text,
    output: Text,
    train_path: Optional[Text] = None,
    kwargs: Optional[Dict] = None,
) -> Optional[Text]:
    """Trains a Core model.

    Args:
        domain: Path to the domain file.
        config: Path to the config file for Core.
        stories: Path to the Core training data.
        output: Output path.
        train_path: If `None` the model will be trained in a temporary
            directory, otherwise in the provided directory.
        kwargs: Additional training parameters.

    Returns:
        If `train_path` is given it returns the path to the model archive,
        otherwise the path to the directory with the trained model files.

    """
    import rasa.core.train

    _train_path = train_path or tempfile.mkdtemp()

    # normal (not compare) training
    core_model = await rasa.core.train(
        domain_file=domain,
        stories_file=stories,
        output_path=os.path.join(_train_path, "core"),
        policy_config=config,
        kwargs=kwargs,
    )

    if not train_path:
        # Only Core was trained.
        stories = data.get_core_directory(stories)
        output_path = create_output_path(output, prefix="core-")
        new_fingerprint = model.model_fingerprint(config, domain, stories=stories)
        model.create_package_rasa(_train_path, output_path, new_fingerprint)
        print_success(
            "Your Rasa Core model is trained and saved at '{}'.".format(output_path)
        )

    return core_model


def train_nlu(
    config: Text, nlu_data: Text, output: Text, train_path: Optional[Text]
) -> Optional["Interpreter"]:
    """Trains a NLU model.

    Args:
        config: Path to the config file for NLU.
        nlu_data: Path to the NLU training data.
        output: Output path.
        train_path: If `None` the model will be trained in a temporary
            directory, otherwise in the provided directory.

    Returns:
        If `train_path` is given it returns the path to the model archive,
        otherwise the path to the directory with the trained model files.

    """
    import rasa.nlu

    _train_path = train_path or tempfile.mkdtemp()
    _, nlu_model, _ = rasa.nlu.train(
        config, nlu_data, _train_path, project="", fixed_model_name="nlu"
    )

    if not train_path:
        nlu_data = data.get_nlu_directory(nlu_data)
        output_path = create_output_path(output, prefix="nlu-")
        new_fingerprint = model.model_fingerprint(config, nlu_data=nlu_data)
        model.create_package_rasa(_train_path, output_path, new_fingerprint)
        print_success(
            "Your Rasa NLU model is trained and saved at '{}'.".format(output_path)
        )

    return nlu_model
