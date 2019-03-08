from rasa_core import constants


def add_run_arguments(parser):
    server_arguments = parser.add_argument_group("Server Settings")
    server_arguments.add_argument(
        '-p', '--port',
        default=constants.DEFAULT_SERVER_PORT,
        type=int,
        help="port to run the server at")
    server_arguments.add_argument(
        '--auth_token',
        type=str,
        help="Enable token based authentication. Requests need to provide "
             "the token to be accepted.")
    server_arguments.add_argument(
        '--cors',
        nargs='*',
        type=str,
        help="enable CORS for the passed origin. "
             "Use * to whitelist all origins")
    server_arguments.add_argument(
        '--enable_api',
        action="store_true",
        help="Start the web server api in addition to the input channel")

    parser.add_argument(
        '-o', '--log_file',
        type=str,
        default="rasa_core.log",
        help="store log file in specified file")
    channel_arguments = parser.add_argument_group("Channels")
    channel_arguments.add_argument(
        '--credentials',
        default=None,
        help="authentication credentials for the connector as a yml file")
    channel_arguments.add_argument(
        '-c', '--connector',
        type=str,
        help="service to connect to")
    parser.add_argument(
        '--endpoints',
        default=None,
        help="Configuration file for the connectors as a yml file")

    jwt_auth = parser.add_argument_group('JWT Authentication')
    jwt_auth.add_argument(
        '--jwt_secret',
        type=str,
        help="Public key for asymmetric JWT methods or shared secret"
             "for symmetric methods. Please also make sure to use "
             "--jwt_method to select the method of the signature, "
             "otherwise this argument will be ignored.")
    jwt_auth.add_argument(
        '--jwt_method',
        type=str,
        default="HS256",
        help="Method used for the signature of the JWT authentication "
             "payload.")
