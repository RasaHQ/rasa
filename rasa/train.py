import asyncio
import os
import tempfile
from typing import Text, Optional, List, Union, Dict

from rasa import model, data
from rasa.core.domain import Domain
from rasa.skill import SkillSelector

from rasa.cli.utils import (
    create_output_path,
    print_success,
    missing_config_keys,
    print_warning,
    get_validated_path,
    print_error,
    bcolors,
    print_color,
)
from rasa.constants import (
    DEFAULT_MODELS_PATH,
    CONFIG_MANDATORY_KEYS,
    CONFIG_MANDATORY_KEYS_CORE,
    CONFIG_MANDATORY_KEYS_NLU,
    FALLBACK_CONFIG_PATH,
)


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
    domain: Optional,
    config: Text,
    training_files: Optional[Union[Text, List[Text]]],
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
    config = get_valid_config(config, CONFIG_MANDATORY_KEYS)

    train_path = tempfile.mkdtemp()
    old_model = model.get_latest_model(output)
    retrain_core = True
    retrain_nlu = True

    skill_imports = SkillSelector.load(config)
    domain = Domain.load(domain, skill_imports)

    story_directory, nlu_data_directory = data.get_core_nlu_directories(
        training_files, skill_imports
    )
    new_fingerprint = model.model_fingerprint(
        config, domain, nlu_data_directory, story_directory
    )

    dialogue_data_not_present = not os.listdir(story_directory)
    nlu_data_not_present = not os.listdir(nlu_data_directory)

    if dialogue_data_not_present and nlu_data_not_present:
        print_error(
            "No training data given. Please provide dialogue and NLU data in "
            "order to train a Rasa model."
        )
        return

    if dialogue_data_not_present:
        print_warning(
            "No dialogue data present. Just a Rasa NLU model will be trained."
        )
        return _train_nlu_with_validated_data(config, nlu_data_directory, output, None)

    if nlu_data_not_present:
        print_warning("No NLU data present. Just a Rasa Core model will be trained.")
        return await _train_core_with_validated_data(
            domain, config, story_directory, output, None, kwargs
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
        await _train_core_with_validated_data(
            domain, config, story_directory, output, train_path, kwargs
        )
    else:
        print (
            "Dialogue data / configuration did not change. "
            "No need to retrain dialogue model."
        )

    if force_training or retrain_nlu:
        _train_nlu_with_validated_data(config, nlu_data_directory, output, train_path)
    else:
        print ("NLU data / configuration did not change. No need to retrain NLU model.")

    if retrain_core or retrain_nlu:
        output = create_output_path(output)
        model.create_package_rasa(train_path, output, new_fingerprint)

        print_success("Your bot is trained and ready to take for a spin!")

        return output
    else:
        print_success(
            "Nothing changed. You can use the old model stored at '{}'"
            "".format(os.path.abspath(old_model))
        )

        return old_model


def train_core(
    domain: Union[Domain, Text],
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
    domain: Union[Domain, Text],
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

    config = get_valid_config(config, CONFIG_MANDATORY_KEYS_CORE)
    skill_imports = SkillSelector.load(config)

    if isinstance(domain, str):
        domain = Domain.load(domain, skill_imports)

    story_directory = data.get_core_directory(stories, skill_imports)

    return await _train_core_with_validated_data(
        domain, config, story_directory, output, train_path, kwargs
    )


async def _train_core_with_validated_data(
    domain: Domain,
    config: Text,
    story_directory: Text,
    output: Text,
    train_path: Optional[Text] = None,
    kwargs: Optional[Dict] = None,
) -> Optional[Text]:
    """Train Core with validated training and config data."""

    import rasa.core.train

    if not os.listdir(story_directory):
        print_error(
            "No dialogue data given. Please provide dialogue data in order to "
            "train a Rasa Core model."
        )
        return

    _train_path = train_path or tempfile.mkdtemp()

    # normal (not compare) training
    print_color("Start training dialogue model ...", color=bcolors.OKBLUE)
    await rasa.core.train(
        domain_file=domain,
        stories_file=story_directory,
        output_path=os.path.join(_train_path, "core"),
        policy_config=config,
        kwargs=kwargs,
    )
    print_color("Done.", color=bcolors.OKBLUE)

    if not train_path:
        # Only Core was trained.
        output_path = create_output_path(output, prefix="core-")
        new_fingerprint = model.model_fingerprint(
            config, domain, stories=story_directory
        )
        model.create_package_rasa(_train_path, output_path, new_fingerprint)
        print_success(
            "Your Rasa Core model is trained and saved at '{}'.".format(output_path)
        )

        return output_path

    return _train_path


def train_nlu(
    config: Text, nlu_data: Text, output: Text, train_path: Optional[Text]
) -> Optional[Text]:
    """Trains an NLU model.

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
    config = get_valid_config(config, CONFIG_MANDATORY_KEYS_NLU)

    # training NLU only hence the training files still have to be selected
    skill_imports = SkillSelector.load(config)
    nlu_data_directory = data.get_nlu_directory(nlu_data, skill_imports)

    return _train_nlu_with_validated_data(
        config, nlu_data_directory, output, train_path
    )


def _train_nlu_with_validated_data(
    config: Text, nlu_data_directory: Text, output: Text, train_path: Optional[Text]
) -> Optional[Text]:
    """Train NLU with validated training and config data."""

    import rasa.nlu.train

    if not os.listdir(nlu_data_directory):
        print_error(
            "No NLU data given. Please provide NLU data in order to train "
            "a Rasa NLU model."
        )
        return

    _train_path = train_path or tempfile.mkdtemp()

    print_color("Start training NLU model ...", color=bcolors.OKBLUE)
    _, nlu_model, _ = rasa.nlu.train(
        config, nlu_data_directory, _train_path, fixed_model_name="nlu"
    )
    print_color("Done.", color=bcolors.OKBLUE)

    if not train_path:
        output_path = create_output_path(output, prefix="nlu-")
        new_fingerprint = model.model_fingerprint(config, nlu_data=nlu_data_directory)
        model.create_package_rasa(_train_path, output_path, new_fingerprint)
        print_success(
            "Your Rasa NLU model is trained and saved at '{}'.".format(output_path)
        )

        return output_path

    return _train_path


def enrich_config(config_path, missing_keys, FALLBACK_CONFIG_PATH):
    import rasa.utils.io

    config_data = rasa.utils.io.read_yaml_file(config_path)
    fallback_config_data = rasa.utils.io.read_yaml_file(FALLBACK_CONFIG_PATH)

    for k in missing_keys:
        config_data[k] = fallback_config_data[k]

    rasa.utils.io.write_yaml_file(config_data, config_path)


def get_valid_config(config: Text, mandatory_keys: List[Text]) -> Text:
    config_path = get_validated_path(config, "config", FALLBACK_CONFIG_PATH)

    missing_keys = missing_config_keys(config_path, mandatory_keys)

    if missing_keys:
        print_warning(
            "Configuration file '{}' is missing mandatory parameters: "
            "{}. Filling missing parameters from fallback configuration file: '{}'."
            "".format(config, ", ".join(missing_keys), FALLBACK_CONFIG_PATH)
        )
        enrich_config(config_path, missing_keys, FALLBACK_CONFIG_PATH)

    return config_path
