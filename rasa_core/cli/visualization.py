def add_visualization_arguments(parser):
    parser.add_argument(
        '-o', '--output',
        default="graph.html",
        type=str,
        help="filename of the output path, e.g. 'graph.html")
    parser.add_argument(
        '-m', '--max_history',
        default=2,
        type=int,
        help="max history to consider when merging "
             "paths in the output graph")
    parser.add_argument(
        '-nlu', '--nlu_data',
        default=None,
        type=str,
        help="path of the Rasa NLU training data, "
             "used to insert example messages into the graph")
