from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import importlib
import pkgutil
from collections import defaultdict

import pytest
from six import PY2


def import_submodules(package_name, skip_list):
    """ Import all submodules of a module, recursively, including subpackages.

    `skip_list` denotes packages that should be skipped during the import"""

    package = importlib.import_module(package_name)
    results = []
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        if full_name not in skip_list:
            module = importlib.import_module(full_name)
            if PY2:
                reload(module)
            else:
                importlib.reload(module)
            results.append(full_name)
            if is_pkg:
                results += import_submodules(full_name, skip_list)
    return results


@pytest.mark.parametrize("banned_package", ["spacy", "mitie", "sklearn", "duckling"])
def test_no_global_imports_of_banned_package(banned_package):
    """This test ensures that neither of the banned packages are imported module wise in any of our code files.

    If one of the dependencies is needed, they should be imported within a function."""
    import inspect

    # To track imports accross modules, we will replace the default import function
    try:
        # noinspection PyCompatibility
        import __builtin__
        original_import_function = __builtin__.__import__
    except ImportError:
        # noinspection PyCompatibility
        import builtins
        original_import_function = builtins.__import__

    tracked_imports = defaultdict(list)

    def import_tracking(name, *x, **xs):
        caller = inspect.currentframe().f_back
        caller_name = caller.f_globals.get('__name__')
        tracked_imports[name].append(caller_name)
        return original_import_function(name, *x, **xs)

    if PY2:
        __builtin__.__import__ = import_tracking
    else:
        builtins.__import__ = import_tracking

    # import all available modules and track imports on the way
    import_submodules("rasa_nlu", skip_list={"rasa_nlu.wsgi"})      # wsgi needs to be skipped as it instantiates an app

    def find_modules_importing(name):
        return {v for k, vs in tracked_imports.items() if k.startswith(name) for v in vs}

    assert not find_modules_importing(banned_package), \
        "No module should import {} globally. Found in {}".format(
            banned_package, ", ".join(find_modules_importing(banned_package)))
