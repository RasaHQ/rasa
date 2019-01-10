import os
import shutil
import tempfile

import questionary
import rasa
import rasa_nlu.train
import rasa_core.train
from rasa_core.utils import print_success, get_file_hash
from rasa_nlu import config


def add_subparser(subparsers):
    scaffold_parser = subparsers.add_parser(
        'train',
        help='Train a Rasa bot')
    scaffold_parser.set_defaults(func=run)


def create_package_rasa(model_dir, output_filename):
    import tarfile

    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(model_dir, arcname=os.path.basename(model_dir))
    return output_filename


def fingerprint(config_file, nlu_file, domain_file, stories_file):
    model_fingerprint = {
        "config": get_file_hash(config_file),
        "nlu_data": get_file_hash(nlu_file),
        "domain": get_file_hash(domain_file),
        "stories": get_file_hash(stories_file),
        "nlu_version": rasa_nlu.__version__,
        "core_version": rasa_core.__version__,
        "version": rasa.__version__
    }


def train(config_file, nlu_file, domain_file, stories_file, output_file):
    train_path = tempfile.mkdtemp()
    rasa_nlu.train.do_train(
        config.load(config_file),
        nlu_file,
        train_path,
        project="rasa_model",
        fixed_model_name="nlu")

    rasa_core.train.train_dialogue_model(
        domain_file=domain_file,
        stories_file=stories_file,
        output_path=os.path.join(train_path, "rasa_model", "core"),
        policy_config=config_file)

    create_package_rasa(os.path.join(train_path, "rasa_model"), output_file)
    shutil.rmtree(train_path)

    print("Train path: {}".format(train_path))

    print_success("Your bot is trained and ready to take for a spin!")


def run(args):
    path = questionary.text("Please enter a folder where the trained model"
                            "gets stored").ask()
