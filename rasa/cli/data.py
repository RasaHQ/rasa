def add_subparser(subparsers, parents):
    from rasa_nlu import convert

    data_parser = subparsers.add_parser(
        "data",
        conflict_handler="resolve",
        parents=parents,
        help="Utils for the Rasa training files.")

    data_subparsers = data_parser.add_subparsers()
    convert_parser = data_subparsers.add_parser(
        "convert", help="Convert Rasa data between different formats.")

    convert_subparsers = convert_parser.add_subparsers()
    convert_nlu_parser = convert_subparsers.add_parser(
        "nlu", help="Convert NLU training data between markdown and json.")

    convert.add_arguments(convert_parser)
    convert_nlu_parser.set_defaults(func=convert.main)
