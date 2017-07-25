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
    parser.add_argument('-o', '--out_file',
                        help='file where to save training data in rasa format')
    parser.add_argument('-f', '--format',
                        help="output format. 'json' or 'md'")
    return parser


def convert_training_data(data_file, out_file, output_format):
    td = load_data(data_file)
    with io.open(out_file, "w") as f:
        if output_format == 'md':
            f.write(td.as_markdown())
        else:
            f.write(td.as_json(indent=2))


if __name__ == "__main__":
    parser = create_argparser()
    args = parser.parse_args()
    convert_training_data(args.data_file, args.out_file, args.format)
