import os


def create_dir_for_file(file_path):
    try:
        os.makedirs(os.path.dirname(file_path))
    except OSError:
        # be happy if someone already created the path
        pass


def recursively_find_files(resource_name):
    """resource_name can be a folder or a file. In both cases we will return a list of files"""
    if not resource_name:
        raise ValueError("Resource name '{}' must be an existing directory or file.".format(resource_name))
    elif os.path.isfile(resource_name):
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


def add_entities_if_synonyms(synonyms_dict, entity_a, entity_b):
    if entity_b is not None:
        original = entity_a.lower() if type(entity_a) == unicode else unicode(entity_a)
        replacement = entity_b.lower() if type(entity_b) == unicode else unicode(entity_b)

        if original != replacement:
            synonyms_dict[original] = replacement
