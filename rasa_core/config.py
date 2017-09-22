import configargparse

core_config_parser = configargparse.ArgParser(
        default_config_files=['sample_configs/config_defaults.yml'],
        config_file_parser_class=configargparse.YAMLConfigFileParser)

core_config_parser.add_argument(
        '-c', '--config',
        required=True, is_config_file=True,
        help='configuration file path')
