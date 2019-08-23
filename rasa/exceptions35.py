# workaround to pass unit tests for Python 3.5
# ModuleNotFoundError was added in Python 3.6
# see PR #4166 for details


class ModuleNotFoundError(Exception):
    pass
