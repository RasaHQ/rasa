
def update_config(config,args,exclude=[],required=[]):
    "override config params with cmd line args, raise err if undefined"
    _args = dict(vars(args))
    params = [k for k in _args.keys() if not k in exclude]
    for param in params:
        replace = (_args.get(param) is not None)
        if (replace):
            config[param] = _args[param]
    for param in required:
        if (config.get(param) is None):
            raise ValueError("parameter {0} unspecified. Please provide a value via the command line or in the config file.".format(param))

    return config