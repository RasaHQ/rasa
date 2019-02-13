import os


def check_path_exists(current, parameter, default=None, none_is_valid=False):
    if not os.path.exists(default) or (not none_is_valid and current is None):
        default_clause = ""
        if default:
            default_clause = ("use the default location ('{}') or "
                              "".format(default))
        print("'{}' does not exist. Please make sure to {}specify it with '{}'."
              "".format(current, default_clause, parameter))
        exit(1)

    return current
