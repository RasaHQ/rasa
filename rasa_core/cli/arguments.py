from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pkg_resources


def add_config_arg(parser):
    """Add an argument to the parser to request a policy configuration."""

    parser.add_argument(
            '-c', '--config',
            type=str,
            nargs="*",
            default=[pkg_resources.resource_filename(__name__,
                                                     "../default_config.yml")],
            help="Policy specification yaml file.")
    return parser


def add_domain_arg(parser):
    """Add an argument to the parser to request a the domain file."""

    parser.add_argument(
            '-d', '--domain',
            type=str,
            required=True,
            help="domain specification yaml file")
    return parser


def add_model_and_story_group(parser, allow_pretrained_model=True):
    """Add an argument to the parser to request a story source."""

    # either the user can pass in a story file, or the data will get
    # downloaded from a url
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
            '-s', '--stories',
            type=str,
            help="file or folder containing the training stories")
    group.add_argument(
            '--url',
            type=str,
            help="If supplied, downloads a story file from a URL and "
                 "trains on it. Fetches the data by sending a GET request "
                 "to the supplied URL.")
    if allow_pretrained_model:
        group.add_argument(
                '--core',
                default=None,
                help="path to load a pre-trained model instead of training ("
                     "for interactive mode only)")
    return parser
