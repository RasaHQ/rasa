import argparse
import json
from rasa_nlu.training_data import TrainingData


def create_argparser():
    parser = argparse.ArgumentParser(description='train a custom language parser')
    parser.add_argument('-d', '--data_file',
                        help='file or dir containing training data')
    parser.add_argument('-b', '--backend', default='default', choices=['mitie', 'spacy_sklearn', 'default'],
                        help='backend to use to tokenize text')
    parser.add_argument('-l', '--language', default=None, choices=['de', 'en'], help="model and data language")
    parser.add_argument('-o', '--out_file',
                        help='file where to save training data in rasa format')
    return parser


def write_file(td, out_file):
    with open(out_file, 'wb') as f:
        f.write(td.as_json(indent=2))

if __name__ == "__main__":
    parser = create_argparser()
    args = parser.parse_args()
    td = TrainingData(args.data_file, args.backend, args.language)
    write_file(td, args.out_file)
