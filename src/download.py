import argparse
import logging
import os
import warnings

from rasa_nlu.config import RasaNLUConfig
from tqdm import tqdm
import requests


def create_argparser():
    parser = argparse.ArgumentParser(description='parse download commands')
    parser.add_argument('-c', '--config',
                        help="config file, all the command line options can also be passed via a (json-formatted) " +
                             "config file. NB command line args take precedence")
    parser.add_argument('-p', '--package',
                        help='package to be downloaded',
                        choices=['mitie'],
                        required=True)

    return parser


def download_mitie_fe_file(fe_file):
    """Download the mitie feature extractor needed to run & train mitie classifiers.

    See https://github.com/mit-nlp/MITIE#initial-setup """

    logging.info("Downloading MITIE feature extractor files")
    _fe_file_url = "https://s3-eu-west-1.amazonaws.com/mitie/total_word_feature_extractor.dat"
    logging.info("Downloading from {}".format(_fe_file_url))
    response = requests.get(_fe_file_url, stream=True)

    with open(fe_file, "wb") as output:
        for data in tqdm(response.iter_content(chunk_size=1024*1024), unit='MB', unit_scale=True):
            output.write(data)
    logging.debug("file written! {0}, {1}".format(fe_file, os.path.exists(fe_file)))


def download(config, pkg="mitie"):
    if pkg == "mitie":
        download_mitie_fe_file(config.mitie_file)
    else:
        warnings.warn("Error. Package {0} not available for download.".format(pkg))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = create_argparser()
    cmdline_args = {key: val for key, val in vars(parser.parse_args()).items() if val is not None}
    config = RasaNLUConfig(cmdline_args.get("config"), os.environ, cmdline_args)
    download(config, cmdline_args["package"])
