import os

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
    
def recursively_find_files(resource_name):
    """resource_name can be a folder or a file. In both cases we will return a list of files"""
    if os.path.isfile(resource_name):
        return [resource_name]        
    elif os.path.isdir(resource_name):
        resources = []
        # walk the fs tree and return a list of files
        nodes_to_visit = [resource_name]
        while(len(nodes_to_visit) > 0):
            # skip hidden files
            nodes_to_visit = filter( lambda f: not f.split("/")[-1].startswith('.'), nodes_to_visit)

            current_node = nodes_to_visit[0]
            # if current node is a folder, schedule its children for a visit. Else add them to the resources.
            if (os.path.isdir(current_node)):
                nodes_to_visit += [os.path.join(current_node, f) for f in os.listdir(current_node)]
            else:
                resources += [current_node]
            nodes_to_visit = nodes_to_visit[1:]
        return resources
    else:
        raise ValueError("resource name must be a dir or a file")
