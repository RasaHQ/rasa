import os
import sys
import argparse
from rasa_nlu.config import RasaNLUConfig


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


def download(config, pkg="mitie"):
    if pkg == "mitie":
        from rasa_nlu.featurizers.mitie_featurizer import MITIEFeaturizer
        MITIEFeaturizer(config.mitie_file)
    else:
        print "Error. Package {0} not available for download.".format(pkg)

if __name__ == '__main__':
    parser = create_argparser()
    cmdline_args = {key: val for key, val in vars(parser.parse_args()).items() if val is not None}
    config = RasaNLUConfig(cmdline_args.get("config"), os.environ, cmdline_args)
    download(config, cmdline_args["package"])
