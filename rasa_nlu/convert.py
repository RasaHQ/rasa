from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import argparse
import io

from rasa_nlu.converters import load_data


def create_argparser():
    parser = argparse.ArgumentParser(description='train a custom language parser')
    parser.add_argument('-d', '--data_file',
                        help='file or dir containing training data')
    parser.add_argument('-b', '--backend', default='tokenize_mitie', choices=['tokenize_mitie', 'tokenize_spacy'],
                        help='backend to use to tokenize text')
    parser.add_argument('-l', '--language', default=None, choices=['de', 'en'], help="model and data language")
    parser.add_argument('-o', '--out_file',
                        help='file where to save training data in rasa format')
    return parser


def write_file(td, out_file):
    with io.open(out_file, "w") as f:
        f.write(td.as_json(indent=2))

if __name__ == "__main__":
    parser = create_argparser()
    args = parser.parse_args()
    td = load_data(args.data_file, args.language, args.backend)
    write_file(td, args.out_file)
