from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import argparse

from rasa_nlu import training_data
from rasa_nlu.utils import write_to_file


def create_argparser():
    parser = argparse.ArgumentParser(description='Convert training data formats '
                                                 'into one another')
    parser.add_argument('-d', '--data_file',
                        required=True,
                        help='file or dir containing training data')
    parser.add_argument('-o', '--out_file',
                        required=True,
                        help='file where to save training data in rasa format')
    parser.add_argument('-l', '--language',
                        default='en',
                        help='language of the data')
    parser.add_argument('-f', '--format',
                        required=True,
                        help="output format. 'json' or 'md'")
    return parser


def convert_training_data(data_file, out_file, output_format, language):
    td = training_data.load_data(data_file, language)
    output = td.as_markdown() if output_format == 'md' else td.as_json(indent=2)
    write_to_file(out_file, output)


if __name__ == "__main__":
    parser = create_argparser()
    args = parser.parse_args()
    convert_training_data(args.data_file, args.out_file, args.format, args.language)
