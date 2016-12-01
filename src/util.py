import os
from rasa_nlu import config_keys


def update_config(init_config, cmdline_args, env_vars, exclude=None, required=None):
    """override config params with cmd line args, default to env vars, & raise err if undefined"""

    if exclude is None:
        exclude = []

    if required is None:
        required = []

    params = [k for k in config_keys if k not in exclude]
    config = init_config.copy()

    for param in params:

        # override with environment variable
        environ_key = "RASA_{0}".format(param.upper())
        replace = env_vars.get(environ_key) is not None
        if replace:
            config[param] = env_vars[environ_key]

        # override with command line arg
        replace = cmdline_args.get(param) is not None
        if replace:
            config[param] = cmdline_args[param]

    # Error checking
    def is_set(x): return config.get(x) is not None

    for param in required:
        if not is_set(param):
            raise ValueError(
                "parameter {0} unspecified. Please provide a value via the command line or in the config file.".format(
                    param))

    if config.get("backend") == 'mitie':
        if not is_set("mitie_file"):
            raise ValueError("backend set to 'mitie' but mitie_file not specified")
        if config.get("language") != 'en':
            raise ValueError("backend set to 'mitie' but language not set to 'en'.")

    return config


def recursively_find_files(resource_name):
    """resource_name can be a folder or a file. In both cases we will return a list of files"""

    if os.path.isfile(resource_name):
        return [resource_name]
    elif os.path.isdir(resource_name):
        resources = []
        # walk the fs tree and return a list of files
        nodes_to_visit = [resource_name]
        while len(nodes_to_visit) > 0:
            # skip hidden files
            nodes_to_visit = filter(lambda f: not f.split("/")[-1].startswith('.'), nodes_to_visit)

            current_node = nodes_to_visit[0]
            # if current node is a folder, schedule its children for a visit. Else add them to the resources.
            if os.path.isdir(current_node):
                nodes_to_visit += [os.path.join(current_node, f) for f in os.listdir(current_node)]
            else:
                resources += [current_node]
            nodes_to_visit = nodes_to_visit[1:]
        return resources
    elif not os.path.exists(resource_name):
        raise ValueError("Could not locate the resource '{}'.".format(os.path.abspath(resource_name)))
    else:
        raise ValueError("Resource name must be an existing directory or file")
