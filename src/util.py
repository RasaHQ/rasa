import os


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
