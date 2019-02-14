import os


def check_path_exists(current, parameter, default=None, none_is_valid=False):
    print(parameter)
    print(not (current is None and none_is_valid))
    if not (current is None and none_is_valid) and not os.path.exists(current):
        if os.path.exists(default):
            print("'{}' not found. Using default location '{}' instead."
                  "".format(current, default))
            current = default
        elif none_is_valid:
            current = None
        else:
            cancel_cause_not_found(current, parameter, default)

    return current


def cancel_cause_not_found(current, parameter, default):
    default_clause = ""
    if default:
        default_clause = ("use the default location ('{}') or "
                          "".format(default))
    print("The path '{}' does not exist. Please make sure to {}specify it "
          "with '--{}'.".format(current, default_clause, parameter))
    exit(1)


def validate(args, params):
    for p in params:
        none_is_valid = False if len(p) == 2 else p[2]
        validated = check_path_exists(getattr(args, p[0]), p[0], p[1],
                                      none_is_valid)
        setattr(args, p[0], validated)
