def add_server_arguments(parser):
    parser.add_argument(
        "-e",
        "--emulate",
        choices=["wit", "luis", "dialogflow"],
        help="which service to emulate (default: None i.e. use"
        " simple built in format)",
    )
    parser.add_argument(
        "-P", "--port", type=int, default=5000, help="port on which to run server"
    )
    parser.add_argument(
        "-t",
        "--token",
        help="auth token. If set, reject requests which don't "
        "provide this token as a query parameter",
    )
    parser.add_argument("-w", "--write", help="file where logs will be saved")
    parser.add_argument(
        "--path",
        required=True,
        help="working directory of the server. Models are"
        "loaded from this directory and trained models "
        "will be saved here.",
    )
    parser.add_argument(
        "--cors",
        nargs="*",
        help="List of domain patterns from where CORS "
        "(cross-origin resource sharing) calls are "
        "allowed. The default value is `[]` which "
        "forbids all CORS requests.",
    )

    parser.add_argument(
        "--max-training-processes",
        type=int,
        default=1,
        help="Number of processes used to handle training "
        "requests. Increasing this value will have a "
        "great impact on memory usage. It is "
        "recommended to keep the default value.",
    )
    parser.add_argument(
        "--endpoints", help="Configuration file for the model server as a yaml file"
    )
    parser.add_argument(
        "--wait-time-between-pulls",
        type=int,
        default=10,
        help="Wait time in seconds between NLU model server queries.",
    )
    parser.add_argument(
        "--response-log",
        help="Directory where logs will be saved "
        "(containing queries and responses)."
        "If set to ``null`` logging will be disabled.",
    )
    parser.add_argument(
        "--storage",
        help="Set the remote location where models are stored. "
        "E.g. on AWS. If nothing is configured, the "
        "server will only serve the models that are "
        "on disk in the configured `path`.",
    )
    parser.add_argument(
        "-c", "--config", help="Default model configuration file used for training."
    )
